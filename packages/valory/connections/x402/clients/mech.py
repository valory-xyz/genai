# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""A ``requests`` session that pays for calls through the mech marketplace.

Drop-in sibling of ``x402_requests``: the caller posts to
``{facilitator_base_url}{upstream_path}`` and the adapter turns each
call into a Safe-signed marketplace request on ``/mech/{api}/{chain}``.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlsplit

import requests
from eth_account import Account
from requests.adapters import HTTPAdapter

from packages.valory.connections.x402.clients.base import PaymentError
from packages.valory.connections.x402.mech_signing import (
    canonical_request_data,
    compute_safe_message_hash,
    derive_request_id,
)

_logger = logging.getLogger(__name__)

# The deadline only bounds replay of a captured body; the facilitator
# clamps it to its own cap.
DEFAULT_REQUEST_TTL_SECS = 120
DEFAULT_NONCE_RETRIES = 2
# Must outlast the facilitator's upstream deadline (120s): an abandoned call is still charged.
DEFAULT_MECH_TIMEOUT: Tuple[float, float] = (10.0, 150.0)
# How many times to wait out a 409 request_in_progress after a timeout,
# and a 409 requester_busy (another call from the same Safe in flight).
DEFAULT_IN_PROGRESS_RETRIES = 3
DEFAULT_BUSY_RETRIES = 3
DEFAULT_INFO_RATE_LIMIT_RETRIES = 3
# Wall-clock cap on one call, all waits and retries included, so a
# struggling facilitator cannot hold the connection's single worker for
# many minutes. One attempt may still run its read timeout past this.
DEFAULT_TOTAL_DEADLINE_SECS = 300.0

_NONCE_MISMATCH = "nonce_mismatch"
_IN_PROGRESS = "request_in_progress"
_BUSY = "requester_busy"
# Answers that leave it unknown whether the facilitator served the call.
_GATEWAY_STATUSES = {502, 504}


class MechDepositRequiredError(PaymentError):
    """The Safe's marketplace pre-deposit cannot cover one more call."""

    def __init__(
        self, *, balance: int, reserved: int, available: int, required: int
    ) -> None:
        """Initialise with the facilitator's 402 context.

        :param balance: on-chain pre-deposit in token base units.
        :param reserved: rates of calls already served but not settled.
        :param available: balance minus reserved.
        :param required: the delivery rate of one call.
        """
        super().__init__(
            f"mech pre-deposit too low: available {available} < required {required}"
        )
        self.balance = balance
        self.reserved = reserved
        self.available = available
        self.required = required


class MechRequestRejectedError(PaymentError):
    """The facilitator refused the request before forwarding it upstream."""

    def __init__(self, *, status_code: int, error: str, detail: str) -> None:
        """Initialise from a facilitator error body.

        :param status_code: HTTP status the facilitator answered with.
        :param error: the facilitator's short error code.
        :param detail: the facilitator's human-readable detail.
        """
        super().__init__(
            f"mech facilitator rejected request ({status_code} {error}): {detail}"
        )
        self.status_code = status_code
        self.error = error
        self.detail = detail


class MechRateExceededError(PaymentError):
    """The facilitator's delivery rate is above the caller's cap."""


class MechDeadlineExceededError(PaymentError):
    """The call's total wall-clock budget ran out across its waits and retries."""


@dataclass(frozen=True)
class RequesterInfo:
    """Parsed ``GET /mech/{chain}/requester/{safe}`` response."""

    chain_id: int
    marketplace_address: str
    mech_address: str
    payment_type: bytes
    delivery_rates: Dict[str, int]
    next_nonce: int
    balance: int
    available: int
    max_ttl_secs: int
    # Facilitator clock (unix seconds) from the response's Date header, so
    # expires_at is not hostage to the agent's clock; None when absent.
    server_time: Optional[int] = None

    @classmethod
    def from_json(
        cls, data: Dict[str, Any], *, server_time: Optional[int] = None
    ) -> "RequesterInfo":
        """Build from the facilitator's JSON body.

        :param data: decoded response body.
        :param server_time: the facilitator's clock at the time of the response.
        :return: the parsed info.
        """
        payment_type = str(data["payment_type"])
        return cls(
            server_time=server_time,
            chain_id=int(data["chain_id"]),
            marketplace_address=str(data["marketplace_address"]),
            mech_address=str(data["mech_address"]),
            payment_type=bytes.fromhex(payment_type[2:]),
            delivery_rates={k: int(v) for k, v in dict(data["delivery_rates"]).items()},
            next_nonce=int(data["next_nonce"]),
            balance=int(data["balance"]),
            available=int(data["available"]),
            max_ttl_secs=int(data["max_ttl_secs"]),
        )


def _server_time(response: requests.Response) -> Optional[int]:
    """Return the ``Date`` header as unix seconds, or ``None`` if absent or malformed."""
    raw = response.headers.get("Date")
    if not raw:
        return None
    try:
        return int(parsedate_to_datetime(raw).timestamp())
    except (TypeError, ValueError):
        return None


# The facilitator reports the remaining reservation window, up to its
# upstream deadline; never wait longer than our own read timeout.
_RETRY_AFTER_CAP_SECS = DEFAULT_MECH_TIMEOUT[1]


def _retry_after_secs(
    response: requests.Response, *, cap: float = _RETRY_AFTER_CAP_SECS
) -> float:
    """Return the ``Retry-After`` header in seconds, bounded to ``[1, cap]``."""
    try:
        wait = float(response.headers.get("Retry-After", "1"))
    except ValueError:
        wait = 1.0
    return max(1.0, min(cap, wait))


def _max_wait_secs(response: requests.Response) -> float:
    """Return the facilitator's ``max_wait_secs`` hint (the most a wait can take), else 0."""
    parsed = _facilitator_error(response)
    if parsed is None:
        return 0.0
    try:
        return float(parsed[2].get("max_wait_secs", 0))
    except (TypeError, ValueError):
        return 0.0


def _wait_budget_secs(response: requests.Response, default_attempts: int) -> float:
    """Total time worth spending on one 409: the hint if given, else a few Retry-After rounds."""
    hinted = _max_wait_secs(response)
    return hinted if hinted > 0 else default_attempts * _retry_after_secs(response)


def _facilitator_error(response: requests.Response) -> Optional[Tuple[str, str, Dict]]:
    """Return ``(error, detail, context)`` if ``response`` is a facilitator error body.

    Facilitator errors carry a top-level ``detail``; upstream answers
    passed through (a Gemini 400, a CoinGecko 429) do not.

    :param response: the facilitator's response.
    :return: the parsed error, or ``None`` for an upstream pass-through.
    """
    try:
        body = response.json()
    except ValueError:
        return None
    if not isinstance(body, dict) or "detail" not in body:
        return None
    detail = body["detail"]
    if isinstance(detail, dict):
        return (
            str(detail.get("error", "error")),
            str(detail.get("detail", "")),
            dict(detail.get("context") or {}),
        )
    return ("error", str(detail), {})


class MechHTTPAdapter(HTTPAdapter):  # pylint: disable=too-many-instance-attributes
    """HTTP adapter that rewrites requests into signed mech-marketplace calls."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        account: Account,
        *,
        safe_address: str,
        chain: str,
        api: str,
        facilitator_base_url: str,
        max_delivery_rate: Optional[int] = None,
        ttl_secs: int = DEFAULT_REQUEST_TTL_SECS,
        nonce_retries: int = DEFAULT_NONCE_RETRIES,
        default_timeout: Union[float, Tuple[float, float]] = DEFAULT_MECH_TIMEOUT,
        total_deadline_secs: float = DEFAULT_TOTAL_DEADLINE_SECS,
        **kwargs: Any,
    ) -> None:
        """Initialise the adapter.

        :param account: the agent EOA, sole owner of ``safe_address``.
        :param safe_address: the service Safe that pays for the calls.
        :param chain: facilitator chain slug, e.g. ``optimism``.
        :param api: facilitator api slug, ``chat`` or ``coingecko``.
        :param facilitator_base_url: origin of the facilitator, no path.
        :param max_delivery_rate: refuse to sign a rate above this (base units).
        :param ttl_secs: how long a signed request stays valid.
        :param nonce_retries: how many nonce collisions to retry through.
        :param default_timeout: timeout applied when the caller passes none.
        :param total_deadline_secs: wall-clock cap on one call including retries.
        :param kwargs: passed to ``HTTPAdapter``.
        """
        super().__init__(**kwargs)
        self.total_deadline_secs = total_deadline_secs
        self.account = account
        self.safe_address = safe_address
        self.chain = chain.lower()
        self.api = api
        self.facilitator_base_url = facilitator_base_url.rstrip("/")
        self.max_delivery_rate = max_delivery_rate
        self.ttl_secs = ttl_secs
        self.nonce_retries = nonce_retries
        self._default_timeout = default_timeout
        self._origin = urlsplit(self.facilitator_base_url)

    def _upstream_call(self, request: requests.PreparedRequest) -> Dict[str, Any]:
        parts = urlsplit(str(request.url))
        if (parts.scheme, parts.netloc) != (self._origin.scheme, self._origin.netloc):
            raise PaymentError(
                f"request host {parts.scheme}://{parts.netloc} is not the "
                f"facilitator {self.facilitator_base_url}"
            )
        body = request.body or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        return {
            "method": (request.method or "GET").upper(),
            "path": parts.path or "/",
            "query": dict(parse_qsl(parts.query, keep_blank_values=True)),
            "body_b64": base64.b64encode(body).decode("ascii") if body else "",
        }

    @staticmethod
    def _wait(deadline: float, secs: float) -> None:
        """Sleep ``secs`` unless that would pass ``deadline``, in which case give up."""
        if secs > deadline - time.monotonic():
            raise MechDeadlineExceededError(
                f"mech call exceeded its {DEFAULT_TOTAL_DEADLINE_SECS:.0f}s budget"
            )
        time.sleep(secs)

    @staticmethod
    def _check(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise MechDeadlineExceededError("mech call exceeded its budget")

    def _fetch_info(self, deadline: float, **send_kwargs: Any) -> RequesterInfo:
        url = f"{self.facilitator_base_url}/mech/{self.chain}/requester/{self.safe_address}"
        prepared = requests.Request("GET", url).prepare()
        response = super().send(prepared, **send_kwargs)
        for _ in range(DEFAULT_INFO_RATE_LIMIT_RETRIES):
            if response.status_code != 429:
                break
            # The route is rate limited process-wide; it says when to retry.
            self._wait(deadline, _retry_after_secs(response))
            response = super().send(prepared, **send_kwargs)
        if response.status_code != 200:
            parsed = _facilitator_error(response)
            error, detail = (
                (parsed[0], parsed[1]) if parsed else ("error", response.text)
            )
            raise MechRequestRejectedError(
                status_code=response.status_code, error=error, detail=detail
            )
        return RequesterInfo.from_json(
            response.json(), server_time=_server_time(response)
        )

    def _signed_body(
        self, call: Dict[str, Any], info: RequesterInfo, nonce: int
    ) -> Dict[str, Any]:
        rate = info.delivery_rates.get(self.api)
        if rate is None:
            raise MechRequestRejectedError(
                status_code=200,
                error="unknown_api",
                detail=f"facilitator has no delivery rate for api {self.api!r}",
            )
        if self.max_delivery_rate is not None and rate > self.max_delivery_rate:
            raise MechRateExceededError(
                f"delivery rate {rate} exceeds cap {self.max_delivery_rate}"
            )
        now = info.server_time if info.server_time is not None else int(time.time())
        expires_at = now + min(self.ttl_secs, info.max_ttl_secs)
        data = canonical_request_data(
            method=call["method"],
            path=call["path"],
            query=call["query"],
            body_b64=call["body_b64"],
            expires_at=expires_at,
        )
        request_id = derive_request_id(
            marketplace_address=info.marketplace_address,
            mech_address=info.mech_address,
            requester=self.safe_address,
            data=data,
            delivery_rate=rate,
            payment_type=info.payment_type,
            nonce=nonce,
            chain_id=info.chain_id,
        )
        digest = compute_safe_message_hash(request_id, self.safe_address, info.chain_id)
        signed = self.account.unsafe_sign_hash(digest)
        return {
            "request_id": "0x" + request_id.hex(),
            "safe_address": self.safe_address,
            "signature": "0x" + bytes(signed.signature).hex(),
            "nonce": str(nonce),
            "expires_at": expires_at,
            "upstream": call,
        }

    def _post_signed(
        self, prepared: requests.PreparedRequest, deadline: float, **kwargs: Any
    ) -> requests.Response:
        """Post a signed body; if the outcome is unknown, ask again for the same body.

        :param prepared: the signed POST.
        :param deadline: monotonic instant after which no more waiting is done.
        :param kwargs: adapter send arguments.
        :return: the facilitator's response.
        """
        try:
            response = super().send(prepared, **kwargs)
            if response.status_code not in _GATEWAY_STATUSES:
                return response
            _logger.warning(
                "mech request answered %s; asking for the stored response",
                response.status_code,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            _logger.warning("mech request failed; asking for the stored response")
        self._check(deadline)
        response = super().send(prepared, **kwargs)
        # The original may still be finishing on the facilitator; it says
        # when to ask again and, at most, how long that could take.
        budget: Optional[float] = None
        while True:
            parsed = _facilitator_error(response)
            if (
                response.status_code != 409
                or parsed is None
                or parsed[0] != _IN_PROGRESS
            ):
                return response
            if budget is None:
                budget = _wait_budget_secs(response, DEFAULT_IN_PROGRESS_RETRIES)
            wait = _retry_after_secs(response)
            if wait > budget:
                return response
            budget -= wait
            self._wait(deadline, wait)
            response = super().send(prepared, **kwargs)

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        """Sign the call for the marketplace and post it to the facilitator.

        :param request: the caller's prepared request to the upstream path.
        :param kwargs: adapter send arguments; ``timeout`` defaults when absent.
        :return: the upstream response as returned by the facilitator.
        """
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._default_timeout
        deadline = time.monotonic() + self.total_deadline_secs

        call = self._upstream_call(request)
        info = self._fetch_info(deadline, **kwargs)
        nonce = info.next_nonce
        url = f"{self.facilitator_base_url}/mech/{self.api}/{self.chain}"
        busy_budget: Optional[float] = None

        attempt = 0
        while attempt <= self.nonce_retries:
            body = self._signed_body(call, info, nonce)
            prepared = requests.Request(
                "POST",
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(body),
            ).prepare()
            self._check(deadline)
            response = self._post_signed(prepared, deadline, **kwargs)
            if 200 <= response.status_code < 300:
                return response

            parsed = _facilitator_error(response)
            if parsed is None:
                return response
            error, detail, context = parsed
            if response.status_code == 402:
                raise MechDepositRequiredError(
                    balance=int(context.get("balance", 0)),
                    reserved=int(context.get("reserved", 0)),
                    available=int(context.get("available", 0)),
                    required=int(context.get("required", 0)),
                )
            if response.status_code == 409 and error == _BUSY:
                # Another call from this Safe is in flight; when it is done
                # the next nonce may have moved, so re-read it.
                if busy_budget is None:
                    busy_budget = _wait_budget_secs(response, DEFAULT_BUSY_RETRIES)
                wait = _retry_after_secs(response)
                if wait <= busy_budget:
                    busy_budget -= wait
                    self._wait(deadline, wait)
                    info = self._fetch_info(deadline, **kwargs)
                    nonce = info.next_nonce
                    continue
            if (
                response.status_code == 409
                and error == _NONCE_MISMATCH
                and attempt < self.nonce_retries
            ):
                nonce = int(context["expected"])
                attempt += 1
                _logger.info(
                    "mech nonce collision, retrying at slot %s (attempt %s)",
                    nonce,
                    attempt,
                )
                continue
            raise MechRequestRejectedError(
                status_code=response.status_code, error=error, detail=detail
            )
        raise MechRequestRejectedError(  # pragma: no cover - loop always returns or raises
            status_code=409, error=_NONCE_MISMATCH, detail="nonce retries exhausted"
        )


def mech_requests(  # pylint: disable=too-many-arguments
    account: Account,
    *,
    safe_address: str,
    chain: str,
    api: str,
    facilitator_base_url: str,
    max_delivery_rate: Optional[int] = None,
    ttl_secs: int = DEFAULT_REQUEST_TTL_SECS,
    default_timeout: Union[float, Tuple[float, float]] = DEFAULT_MECH_TIMEOUT,
    **kwargs: Any,
) -> requests.Session:
    """Create a requests session that pays for calls through the mech marketplace.

    Requests must target ``{facilitator_base_url}{upstream_path}``; the
    path and query relative to the facilitator origin are forwarded as
    the upstream call.

    :param account: the agent EOA, sole owner of ``safe_address``.
    :param safe_address: the service Safe that pays for the calls.
    :param chain: facilitator chain slug, e.g. ``optimism``.
    :param api: facilitator api slug, ``chat`` or ``coingecko``.
    :param facilitator_base_url: origin of the facilitator, no path.
    :param max_delivery_rate: refuse to sign a rate above this (base units).
    :param ttl_secs: how long a signed request stays valid.
    :param default_timeout: timeout applied when the caller passes none.
    :param kwargs: passed to ``HTTPAdapter``.
    :return: a session with the mech adapter mounted for http and https.
    """
    session = requests.Session()
    adapter = MechHTTPAdapter(
        account,
        safe_address=safe_address,
        chain=chain,
        api=api,
        facilitator_base_url=facilitator_base_url,
        max_delivery_rate=max_delivery_rate,
        ttl_secs=ttl_secs,
        default_timeout=default_timeout,
        **kwargs,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
