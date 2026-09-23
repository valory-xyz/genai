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

Drop-in sibling of ``x402_requests``. The caller keeps posting to
``{facilitator_base_url}{upstream_path}`` exactly as it does with the
x402 proxy; the adapter turns each request into a Safe-signed
marketplace request and posts it to the facilitator's
``/mech/{api}/{chain}`` route, which forwards the call upstream and
settles it later in a batch.

Per request the adapter:

1. asks the facilitator for the requester's current parameters
   (contracts, delivery rate, next nonce, balance), so the agent carries
   no chain configuration and no RPC of its own;
2. builds the canonical bytes of the call with an ``expires_at``,
   derives the marketplace request_id and signs its Safe-message wrap
   with the agent EOA, the Safe's sole owner;
3. posts the signed body and returns the upstream response as-is.

A nonce collision (another in-flight call from the same Safe) is a 409
that names the slot to use; the adapter re-signs at that slot and
retries a bounded number of times. A 402 raises
``MechDepositRequiredError`` so the caller can trigger a top-up.
"""

import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import parse_qsl, urlsplit

import requests
from eth_account import Account
from requests.adapters import HTTPAdapter

from packages.valory.connections.x402.clients.base import PaymentError
from packages.valory.connections.x402.clients.requests import DEFAULT_X402_TIMEOUT
from packages.valory.connections.x402.mech_signing import (
    canonical_request_data,
    compute_safe_message_hash,
    derive_request_id,
)

_logger = logging.getLogger(__name__)

# Well under the facilitator's default cap (300s); the request is served
# within seconds, the deadline only bounds replay of a captured body.
DEFAULT_REQUEST_TTL_SECS = 120
DEFAULT_NONCE_RETRIES = 2

_NONCE_MISMATCH = "nonce_mismatch"


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

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "RequesterInfo":
        """Build from the facilitator's JSON body.

        :param data: decoded response body.
        :return: the parsed info.
        """
        payment_type = str(data["payment_type"])
        return cls(
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
        default_timeout: Union[float, Tuple[float, float]] = DEFAULT_X402_TIMEOUT,
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
        :param kwargs: passed to ``HTTPAdapter``.
        """
        super().__init__(**kwargs)
        self.account = account
        self.safe_address = safe_address
        self.chain = chain
        self.api = api
        self.facilitator_base_url = facilitator_base_url.rstrip("/")
        self.max_delivery_rate = max_delivery_rate
        self.ttl_secs = ttl_secs
        self.nonce_retries = nonce_retries
        self._default_timeout = default_timeout
        self._origin = urlsplit(self.facilitator_base_url)

    # ---------- request assembly -------------------------------------------

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

    def _fetch_info(self, **send_kwargs: Any) -> RequesterInfo:
        url = f"{self.facilitator_base_url}/mech/{self.chain}/requester/{self.safe_address}"
        prepared = requests.Request("GET", url).prepare()
        response = super().send(prepared, **send_kwargs)
        if response.status_code != 200:
            parsed = _facilitator_error(response)
            error, detail = (
                (parsed[0], parsed[1]) if parsed else ("error", response.text)
            )
            raise MechRequestRejectedError(
                status_code=response.status_code, error=error, detail=detail
            )
        return RequesterInfo.from_json(response.json())

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
        expires_at = int(time.time()) + min(self.ttl_secs, info.max_ttl_secs)
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

    # ---------- send -------------------------------------------------------

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        """Sign the call for the marketplace and post it to the facilitator.

        :param request: the caller's prepared request to the upstream path.
        :param kwargs: adapter send arguments; ``timeout`` defaults when absent.
        :return: the upstream response as returned by the facilitator.
        """
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._default_timeout

        call = self._upstream_call(request)
        info = self._fetch_info(**kwargs)
        nonce = info.next_nonce
        url = f"{self.facilitator_base_url}/mech/{self.api}/{self.chain}"

        for attempt in range(self.nonce_retries + 1):
            body = self._signed_body(call, info, nonce)
            prepared = requests.Request(
                "POST",
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(body),
            ).prepare()
            response = super().send(prepared, **kwargs)
            if 200 <= response.status_code < 300:
                return response

            parsed = _facilitator_error(response)
            if parsed is None:
                # An upstream answer passed through unchanged (e.g. a 429
                # from CoinGecko); the caller handles it as it does today.
                return response
            error, detail, context = parsed
            if response.status_code == 402:
                raise MechDepositRequiredError(
                    balance=int(context.get("balance", 0)),
                    reserved=int(context.get("reserved", 0)),
                    available=int(context.get("available", 0)),
                    required=int(context.get("required", 0)),
                )
            if (
                response.status_code == 409
                and error == _NONCE_MISMATCH
                and attempt < self.nonce_retries
            ):
                nonce = int(context["expected"])
                _logger.info(
                    "mech nonce collision, retrying at slot %s (attempt %s)",
                    nonce,
                    attempt + 1,
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
    default_timeout: Union[float, Tuple[float, float]] = DEFAULT_X402_TIMEOUT,
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
