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

"""A ``requests`` session that pays for calls as standard off-chain mech requests.

Drop-in sibling of ``mech_requests``: the caller still sends the upstream
call (a Gemini ``generateContent`` POST, a CoinGecko GET) to
``{facilitator_base_url}{upstream_path}``. The adapter turns it into the
request mech-client sends to any marketplace mech: a tool and a prompt in
signed request metadata, posted to
``{facilitator_base_url}/mech/offchain/{chain}/send_signed_requests``, and
the answer read from ``fetch_offchain_info``. The facilitator mech is
addressed directly, so the request id, nonce and signed limit are all the
requester's, as for any other mech.

The signed ``delivery_rate`` is a limit: the facilitator charges what the
call cost, at most that. It must be at least the tool's minimum, or the
request is refused before anything runs.
"""

import json
import logging
import secrets
import threading
import time
import uuid
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlsplit

import requests
from eth_account import Account
from eth_utils import to_checksum_address
from requests.adapters import HTTPAdapter

from packages.valory.connections.x402.clients.base import PaymentError
from packages.valory.connections.x402.clients.mech import (
    EXPLICIT_NONCE_BITS,
    EXPLICIT_NONCE_DEADLINE_SECS,
    EXPLICIT_NONCE_DISABLED_REASON,
    NONCE_MODE_EXPLICIT,
    NONCE_MODE_SEQUENTIAL,
    ExplicitNonceRefusals,
    MechDeadlineExceededError,
    MechDepositRequiredError,
    MechOutcomeUnknownError,
    MechRequestRejectedError,
    NonceMode,
)
from packages.valory.connections.x402.mech_signing import (
    cidv1_digest,
    compute_explicit_nonce_domain_separator,
    compute_safe_message_hash,
    derive_explicit_request_id,
    derive_request_id,
)

_logger = logging.getLogger(__name__)

TOOL_CHAT = "chat"
TOOL_COINGECKO = "coingecko"

# Marketplace payment types (``IMech.paymentType``): the chain's own coin,
# and USDC. Both are fixed when a mech is created.
PAYMENT_TYPE_NATIVE = bytes.fromhex(
    "ba699a34be8fe0e7725e93dcbce1701b0211a8ca61330aaeb8a05bf2ec7abed1"
)
PAYMENT_TYPE_USDC = bytes.fromhex(
    "6406bb5f31a732f898e1ce9fdd988a80a808d36ab5d9a4a4805a8be8d197d5e3"
)
_PAYMENT_TYPE_DECIMALS = {PAYMENT_TYPE_NATIVE: 18, PAYMENT_TYPE_USDC: 6}

# Per-request limits in millionths of a dollar, converted to the mech's
# payment asset (one unit to the dollar). The default is twice the
# facilitator's expected price, as mech-client signs twice the published
# expected cost; the minimum is what the facilitator refuses below.
# chat: the facilitator's minimum cap is 1.5 cents; it charges tokens used.
# coingecko: its fixed price per call, 0.001, is also its minimum cap.
_MICRO_USD = 6
DEFAULT_LIMITS_MICRO_USD: Dict[str, int] = {TOOL_CHAT: 30_000, TOOL_COINGECKO: 2_000}
MINIMUM_LIMITS_MICRO_USD: Dict[str, int] = {TOOL_CHAT: 15_000, TOOL_COINGECKO: 1_000}


@dataclass(frozen=True)
class FacilitatorMech:
    """The marketplace mech the facilitator settles as, on one chain."""

    chain_id: int
    marketplace_address: str
    mech_address: str
    payment_type: bytes


# Verified on chain: ``mechMarketplace()`` and ``paymentType()`` of each mech.
FACILITATOR_MECHS: Dict[str, FacilitatorMech] = {
    "gnosis": FacilitatorMech(
        chain_id=100,
        marketplace_address="0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB",
        mech_address="0x0E43bEF3472DB83D7418f76947F64159748B78a7",
        payment_type=PAYMENT_TYPE_NATIVE,
    ),
    "polygon": FacilitatorMech(
        chain_id=137,
        marketplace_address="0x343F2B005cF6D70bA610CD9F1F1927049414B582",
        mech_address="0x7390dD0cDB8ae7d3744B409AB1b9326417Aa11E1",
        payment_type=PAYMENT_TYPE_USDC,
    ),
    "optimism": FacilitatorMech(
        chain_id=10,
        marketplace_address="0x46C0D07F55d4F9B5Eed2Fc9680B5953e5fd7b461",
        mech_address="0xcB55661502c8AB004Ab79C58974f1284d803b3Bf",
        payment_type=PAYMENT_TYPE_USDC,
    ),
    "base": FacilitatorMech(
        chain_id=8453,
        marketplace_address="0xf24eE42edA0fc9b33B7D41B06Ee8ccD2Ef7C5020",
        mech_address="0x131876A3542D0945f57886C59BE3dF55D651c8d6",
        payment_type=PAYMENT_TYPE_USDC,
    ),
}


def limit_in_base_units(micro_usd: int, payment_type: bytes) -> int:
    """Return a dollar amount in base units of the mech's payment asset.

    :param micro_usd: the amount in millionths of a dollar.
    :param payment_type: the mech's 32-byte payment type.
    :return: the amount in the asset's base units.
    :raises ValueError: for a payment type whose asset is not known.
    """
    decimals = _PAYMENT_TYPE_DECIMALS.get(bytes(payment_type))
    if decimals is None:
        raise ValueError(f"payment type 0x{bytes(payment_type).hex()} is not priced")
    return micro_usd * 10 ** (decimals - _MICRO_USD)


def default_limit(tool: str, payment_type: bytes) -> int:
    """Return the limit signed per request when none is configured.

    :param tool: ``chat`` or ``coingecko``.
    :param payment_type: the mech's 32-byte payment type.
    :return: the limit in the payment asset's base units.
    """
    return limit_in_base_units(DEFAULT_LIMITS_MICRO_USD[tool], payment_type)


def minimum_limit(tool: str, payment_type: bytes) -> int:
    """Return the least limit the facilitator accepts for ``tool``.

    :param tool: ``chat`` or ``coingecko``.
    :param payment_type: the mech's 32-byte payment type.
    :return: the minimum in the payment asset's base units.
    """
    return limit_in_base_units(MINIMUM_LIMITS_MICRO_USD[tool], payment_type)


# The facilitator holds a poll open up to 20 s, below mech-client's 30 s.
DEFAULT_TIMEOUT: Tuple[float, float] = (10.0, 30.0)
DEFAULT_TOTAL_DEADLINE_SECS = 300.0
# Least time between two polls that came back empty at once, for a
# facilitator that answers without holding the poll.
DEFAULT_POLL_INTERVAL_SECS = 2.0
# Re-sends of one signed form whose answer was lost, and waits on a
# facilitator that says to try again later.
DEFAULT_POST_RETRIES = 2
DEFAULT_BUSY_RETRIES = 5
# Re-signs at the nonce the facilitator says it expects (sequential only).
DEFAULT_NONCE_RETRIES = 3
_RETRY_AFTER_CAP_SECS = 30.0
_GATEWAY_STATUSES = {502, 504}

# The facilitator's reasons, verbatim, in a rejection body's ``reason``.
REASON_BELOW_MINIMUM = "signed cap below the mech's minimum charge"
REASON_INSUFFICIENT_BALANCE = "insufficient balance"
REASON_NONCE_BELOW = "wire nonce below sender's next expected slot"
REASON_NONCE_ABOVE = "wire nonce above sender's next expected slot"
REASON_DEADLINE_TOO_SOON = "explicit-nonce deadline is too close for the mech to settle"
_NONCE_REASONS = {REASON_NONCE_BELOW, REASON_NONCE_ABOVE}

# Who caused a failed run, as the delivered response names it.
FAULT_CLIENT = "client"
FAULT_MECH = "mech"

# Request ids of signed forms whose outcome is unknown, kept so the next
# identical call asks about the same request instead of paying again.
_UNRESOLVED_TTL_SECS = 600.0
_UNRESOLVED_MAX = 32

_REFUSALS = ExplicitNonceRefusals()
# The next sequential nonce per (chain, Safe), learnt from the facilitator.
_NEXT_SEQUENTIAL: Dict[Tuple[str, str], int] = {}
_SEQUENTIAL_GUARD = threading.Lock()
_SAFE_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}


def standard_explicit_nonce_refusals() -> ExplicitNonceRefusals:
    """Return the process-wide record of explicit-nonce refusals on this route.

    :return: the record every standard adapter in this process shares.
    """
    return _REFUSALS


def reset_sequential_nonces() -> None:
    """Forget every learnt sequential nonce, for a process that wants a fresh start."""
    with _SEQUENTIAL_GUARD:
        _NEXT_SEQUENTIAL.clear()


def _sequential_lock(chain: str, safe_address: str) -> threading.Lock:
    """Return the process-wide lock held while a sequential nonce is in flight."""
    key = (chain.lower(), safe_address.lower())
    with _SEQUENTIAL_GUARD:
        lock = _SAFE_LOCKS.get(key)
        if lock is None:
            lock = _SAFE_LOCKS[key] = threading.Lock()
        return lock


class MechLimitBelowMinimumError(PaymentError):
    """The signed limit is below what the facilitator charges at least for the tool."""

    def __init__(self, *, limit: int, minimum: int) -> None:
        """Initialise with both amounts.

        :param limit: the limit that was signed, in base units.
        :param minimum: the facilitator's minimum for the tool, in base units.
        """
        super().__init__(
            f"mech request limit {limit} is below the facilitator's minimum {minimum}"
        )
        self.limit = limit
        self.minimum = minimum


class MechStandardRouteUnavailableError(MechRequestRejectedError):
    """The facilitator does not serve the standard route on this chain.

    Nothing was accepted, so the same call may be sent by another route.
    """


class MechRequestFailedError(PaymentError):
    """The facilitator ran the request and it failed; it is charged all the same.

    ``fault`` is ``client`` when another request would have avoided the
    failure (charged the limit), ``mech`` when it was the facilitator's
    (charged its minimum charge).
    """

    def __init__(
        self, *, request_id: int, fault: str, error: str, delivery_rate: int
    ) -> None:
        """Initialise from a ``failed`` poll answer.

        :param request_id: the marketplace request id.
        :param fault: ``client`` or ``mech``.
        :param error: the facilitator's description.
        :param delivery_rate: what the request was charged, in base units.
        """
        super().__init__(f"mech request failed ({fault} fault): {error}")
        self.request_id = request_id
        self.fault = fault
        self.error = error
        self.delivery_rate = delivery_rate


def _json_object(response: requests.Response) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _retry_after_secs(response: requests.Response) -> float:
    """Return ``Retry-After`` in seconds, bounded to ``[1, _RETRY_AFTER_CAP_SECS]``."""
    try:
        wait = float(response.headers.get("Retry-After", "1"))
    except ValueError:
        wait = 1.0
    return max(1.0, min(_RETRY_AFTER_CAP_SECS, wait))


def _server_time(response: requests.Response) -> Optional[int]:
    raw = response.headers.get("Date")
    if not raw:
        return None
    try:
        return int(parsedate_to_datetime(raw).timestamp())
    except (TypeError, ValueError):
        return None


def _rejection(response: requests.Response) -> MechRequestRejectedError:
    body = _json_object(response)
    reason = body.get("reason") or body.get("detail") or response.text
    return MechRequestRejectedError(
        status_code=response.status_code,
        error=str(body.get("error") or body.get("status") or "rejected"),
        detail=str(reason),
    )


def _deposit_error(body: Dict[str, Any], limit: int) -> MechDepositRequiredError:
    """Build the typed 402 error from the facilitator's payment challenge.

    ``required`` there is this request's limit plus the limits of the
    Safe's requests not yet settled, and ``currentBalance`` the deposit.
    """
    balance = int(body.get("currentBalance") or 0)
    held_with_this = int(body.get("required") or 0)
    reserved = max(0, held_with_this - limit)
    return MechDepositRequiredError(
        balance=balance,
        reserved=reserved,
        available=balance - reserved,
        required=limit,
    )


@dataclass
class _Pending:
    """A signed form the facilitator may hold, and its request id."""

    form: Dict[str, str]
    request_id: int
    expires_at: float


class MechStandardHTTPAdapter(
    HTTPAdapter
):  # pylint: disable=too-many-instance-attributes
    """HTTP adapter that turns upstream calls into standard off-chain mech requests."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        account: Account,
        *,
        safe_address: str,
        chain: str,
        api: str,
        facilitator_base_url: str,
        delivery_rate_limit: Optional[int] = None,
        mech_address: Optional[str] = None,
        payment_type: Optional[bytes] = None,
        nonce_mode: NonceMode = NONCE_MODE_EXPLICIT,
        explicit_nonce_deadline_secs: int = EXPLICIT_NONCE_DEADLINE_SECS,
        default_timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
        total_deadline_secs: float = DEFAULT_TOTAL_DEADLINE_SECS,
        poll_interval_secs: float = DEFAULT_POLL_INTERVAL_SECS,
        **kwargs: Any,
    ) -> None:
        """Initialise the adapter.

        :param account: the agent EOA, sole owner of ``safe_address``.
        :param safe_address: the Safe that pays, the requester of record.
        :param chain: chain name, one of ``FACILITATOR_MECHS``.
        :param api: the tool, ``chat`` or ``coingecko``.
        :param facilitator_base_url: origin of the facilitator, no path.
        :param delivery_rate_limit: the most one request may be charged, in
            the payment asset's base units; ``None`` signs ``default_limit``.
        :param mech_address: the facilitator mech; ``None`` uses the default
            for ``chain``. Another mech needs its ``payment_type`` too.
        :param payment_type: the mech's 32-byte payment type; ``None`` uses
            the default mech's.
        :param nonce_mode: ``explicit`` or ``sequential``. An explicit adapter
            signs sequentially while a refusal from the facilitator is
            remembered, see ``ExplicitNonceRefusals``.
        :param explicit_nonce_deadline_secs: how long past the facilitator's
            clock an explicit request stays settleable.
        :param default_timeout: connect and read timeout of each HTTP call.
        :param total_deadline_secs: wall-clock cap on one call, polls included.
        :param poll_interval_secs: least time between two empty polls.
        :param kwargs: passed to ``HTTPAdapter``.
        :raises ValueError: for an unknown chain, tool or nonce mode, a mech
            without a payment type, or a limit below the tool's minimum.
        """
        if api not in DEFAULT_LIMITS_MICRO_USD:
            raise ValueError(f"unknown mech tool {api!r}")
        if nonce_mode not in (NONCE_MODE_EXPLICIT, NONCE_MODE_SEQUENTIAL):
            raise ValueError(f"unknown nonce_mode {nonce_mode!r}")
        known = FACILITATOR_MECHS.get(chain.lower())
        if known is None:
            raise ValueError(f"no facilitator mech is known on chain {chain!r}")
        mech = to_checksum_address(mech_address or known.mech_address)
        if payment_type is None:
            if mech != to_checksum_address(known.mech_address):
                raise ValueError(
                    f"mech {mech} is not the default facilitator mech on {chain}; "
                    "its payment type must be configured"
                )
            payment_type = known.payment_type
        if len(payment_type) != 32:
            raise ValueError("payment_type must be 32 bytes")
        limit = (
            default_limit(api, payment_type)
            if delivery_rate_limit is None
            else int(delivery_rate_limit)
        )
        minimum = minimum_limit(api, payment_type)
        if limit < minimum:
            raise ValueError(
                f"mech {api} limit {limit} is below the facilitator's minimum {minimum}"
            )
        super().__init__(**kwargs)
        self.account = account
        self.safe_address = to_checksum_address(safe_address)
        self.chain = chain.lower()
        self.tool = api
        self.facilitator_base_url = facilitator_base_url.rstrip("/")
        self.offchain_url = f"{self.facilitator_base_url}/mech/offchain/{self.chain}"
        self.chain_id = known.chain_id
        self.marketplace_address = known.marketplace_address
        self.mech_address = mech
        self.payment_type = bytes(payment_type)
        self.delivery_rate_limit = limit
        self.nonce_mode: NonceMode = nonce_mode
        self.explicit_nonce_deadline_secs = int(explicit_nonce_deadline_secs)
        self.default_timeout = default_timeout
        self.total_deadline_secs = total_deadline_secs
        self.poll_interval_secs = poll_interval_secs
        self._origin = urlsplit(self.facilitator_base_url)
        # Facilitator clock minus ours, from the last Date header seen.
        self._clock_offset = 0
        self._unresolved: Dict[str, _Pending] = {}

    # -- the upstream call as a prompt -----------------------------------

    def _prompt(self, request: requests.PreparedRequest) -> str:
        """Return the prompt the tool serves for the caller's upstream call.

        :param request: the caller's request to ``{base}{upstream_path}``.
        :return: the prompt, as the facilitator's tool reads it.
        :raises PaymentError: for a host other than the facilitator, or a
            call the tool does not serve.
        """
        parts = urlsplit(str(request.url))
        if (parts.scheme, parts.netloc) != (self._origin.scheme, self._origin.netloc):
            raise PaymentError(
                f"request host {parts.scheme}://{parts.netloc} is not the "
                f"facilitator {self.facilitator_base_url}"
            )
        path = parts.path.strip("/")
        if self.tool == TOOL_COINGECKO:
            if (request.method or "GET").upper() != "GET":
                raise PaymentError("coingecko calls must be GET")
            query = dict(parse_qsl(parts.query, keep_blank_values=True))
            return json.dumps({"path": path, "query": query}, sort_keys=True)
        # chat: POST .../models/{model}:generateContent with a JSON body.
        marker = "models/"
        if (request.method or "").upper() != "POST" or marker not in path:
            raise PaymentError(f"chat calls must POST a generateContent path: {path}")
        model, _, method = path.rsplit(marker, 1)[1].partition(":")
        if method != "generateContent" or not model:
            raise PaymentError(f"chat calls must POST a generateContent path: {path}")
        raw = request.body or b"{}"
        try:
            body = json.loads(raw)
        except ValueError as exc:
            raise PaymentError("chat body is not JSON") from exc
        if not isinstance(body, dict):
            raise PaymentError("chat body must be a JSON object")
        return json.dumps({**body, "model": model}, sort_keys=True)

    # -- HTTP helpers ----------------------------------------------------

    def _now(self) -> int:
        """Return the facilitator's clock, as far as its Date headers tell."""
        return int(time.time()) + self._clock_offset

    def _note_clock(self, response: requests.Response) -> None:
        server = _server_time(response)
        if server is not None:
            self._clock_offset = server - int(time.time())

    def _http(
        self,
        method: str,
        url: str,
        deadline: float,
        *,
        form: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> requests.Response:
        """Send one HTTP call with its read timeout clamped to what is left."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MechDeadlineExceededError("mech call exceeded its budget")
        prepared = requests.Request(method, url, data=form, params=params).prepare()
        connect, read = self.default_timeout
        kwargs = dict(kwargs)
        kwargs["timeout"] = (connect, min(read, remaining))
        response = super().send(prepared, **kwargs)
        self._note_clock(response)
        return response

    @staticmethod
    def _wait(deadline: float, secs: float) -> None:
        if secs >= deadline - time.monotonic():
            raise MechDeadlineExceededError("mech call exceeded its budget")
        time.sleep(secs)

    # -- signing ---------------------------------------------------------

    def _form(
        self, ipfs_data: str, nonce: int, settle_by: Optional[int]
    ) -> Tuple[Dict[str, str], int]:
        """Sign one request and return its form body and request id.

        :param ipfs_data: the request metadata, exactly as posted.
        :param nonce: the sequential slot or the chosen explicit nonce.
        :param settle_by: an explicit request's deadline; ``None`` signs the
            sequential id.
        :return: the form fields and the request id as an integer.
        """
        data = cidv1_digest(ipfs_data.encode("utf-8"))
        if settle_by is None:
            request_id = derive_request_id(
                marketplace_address=self.marketplace_address,
                mech_address=self.mech_address,
                requester=self.safe_address,
                data=data,
                delivery_rate=self.delivery_rate_limit,
                payment_type=self.payment_type,
                nonce=nonce,
                chain_id=self.chain_id,
            )
        else:
            request_id = derive_explicit_request_id(
                domain_separator=compute_explicit_nonce_domain_separator(
                    self.chain_id, self.marketplace_address
                ),
                mech_address=self.mech_address,
                requester=self.safe_address,
                data=data,
                max_delivery_rate=self.delivery_rate_limit,
                payment_type=self.payment_type,
                nonce=nonce,
                deadline=settle_by,
            )
        digest = compute_safe_message_hash(request_id, self.safe_address, self.chain_id)
        signed = self.account.unsafe_sign_hash(digest)
        request_id_int = int.from_bytes(request_id, "big")
        form = {
            "sender": self.safe_address,
            "signature": "0x" + bytes(signed.signature).hex(),
            "ipfs_hash": "0x" + data.hex(),
            "request_id": str(request_id_int),
            "delivery_rate": str(self.delivery_rate_limit),
            "nonce": str(nonce),
            "ipfs_data": ipfs_data,
        }
        if settle_by is not None:
            form["nonce_mode"] = NONCE_MODE_EXPLICIT
            form["deadline"] = str(settle_by)
        return form, request_id_int

    # -- posting ---------------------------------------------------------

    def _post(
        self, form: Dict[str, str], deadline: float, **kwargs: Any
    ) -> requests.Response:
        """Post one signed form, re-sending it when the answer is lost or says wait.

        The same form is sent each time, so a request the facilitator
        already took is answered as accepted rather than taken twice.

        :raises MechOutcomeUnknownError: when no answer arrives at all.
        """
        url = f"{self.offchain_url}/send_signed_requests"
        lost = 0
        busy = 0
        while True:
            try:
                response = self._http("POST", url, deadline, form=form, **kwargs)
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                lost += 1
                if lost > DEFAULT_POST_RETRIES:
                    raise MechOutcomeUnknownError(
                        "mech request was sent but no answer arrived: "
                        f"{type(exc).__name__}"
                    ) from exc
                continue
            if response.status_code in _GATEWAY_STATUSES:
                lost += 1
                if lost > DEFAULT_POST_RETRIES:
                    raise MechOutcomeUnknownError(
                        f"mech request was sent and answered {response.status_code}"
                    )
                continue
            if self._says_wait(response) and busy < DEFAULT_BUSY_RETRIES:
                busy += 1
                self._wait(deadline, _retry_after_secs(response))
                continue
            return response

    @staticmethod
    def _says_wait(response: requests.Response) -> bool:
        """Whether the facilitator asks to try the same request again later."""
        if response.status_code == 429:
            return True
        if response.status_code != 503:
            return False
        reason = _json_object(response).get("reason")
        return reason not in (EXPLICIT_NONCE_DISABLED_REASON, REASON_NONCE_ABOVE)

    def _check_accepted(self, response: requests.Response, request_id: int) -> None:
        """Raise the typed error for anything but an acceptance of ``request_id``."""
        body = _json_object(response)
        if response.status_code == 200:
            if str(body.get("request_id")) != str(request_id):
                raise MechRequestRejectedError(
                    status_code=200,
                    error="request_id_mismatch",
                    detail=f"facilitator accepted {body.get('request_id')!r}",
                )
            return
        if response.status_code == 402:
            raise _deposit_error(body, self.delivery_rate_limit)
        if response.status_code == 400 and body.get("reason") == REASON_BELOW_MINIMUM:
            raise MechLimitBelowMinimumError(
                limit=self.delivery_rate_limit,
                minimum=int(body.get("minimum_cap") or 0),
            )
        if response.status_code == 404:
            rejection = _rejection(response)
            raise MechStandardRouteUnavailableError(
                status_code=404, error=rejection.error, detail=rejection.detail
            )
        raise _rejection(response)

    def _submit_explicit(
        self, call_key: str, ipfs_data: str, deadline: float, **kwargs: Any
    ) -> Optional[int]:
        """Send with an explicit nonce; ``None`` when the facilitator turns that away."""
        resigned = False
        while True:
            settle_by = self._now() + self.explicit_nonce_deadline_secs
            form, request_id = self._form(
                ipfs_data, secrets.randbits(EXPLICIT_NONCE_BITS), settle_by
            )
            self._remember(call_key, form, request_id)
            response = self._post(form, deadline, **kwargs)
            body = _json_object(response)
            reason = body.get("reason")
            if response.status_code == 503 and reason == EXPLICIT_NONCE_DISABLED_REASON:
                self._forget(call_key)
                return None
            if (
                response.status_code == 401
                and reason == REASON_DEADLINE_TOO_SOON
                and not resigned
            ):
                # Its clock is ahead of the last Date we saw; sign from its own.
                now = (body.get("context") or {}).get("now")
                if now is not None:
                    self._clock_offset = int(now) - int(time.time())
                resigned = True
                continue
            self._check_accepted(response, request_id)
            return request_id

    def _submit_sequential(
        self, call_key: str, ipfs_data: str, deadline: float, **kwargs: Any
    ) -> int:
        """Send at the Safe's next sequential nonce, as the facilitator reports it.

        The nonce starts from what this process last saw accepted and moves
        to the facilitator's ``expected_nonce`` when it names another, so no
        RPC read is needed. The lock keeps two calls from signing one slot.
        """
        key = (self.chain, self.safe_address.lower())
        lock = _sequential_lock(self.chain, self.safe_address)
        if not lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise MechDeadlineExceededError(
                "mech call gave up waiting for another request on this Safe"
            )
        try:
            with _SEQUENTIAL_GUARD:
                nonce = _NEXT_SEQUENTIAL.get(key, 0)
            for _ in range(DEFAULT_NONCE_RETRIES + 1):
                form, request_id = self._form(ipfs_data, nonce, None)
                self._remember(call_key, form, request_id)
                response = self._post(form, deadline, **kwargs)
                body = _json_object(response)
                expected = body.get("expected_nonce")
                if body.get("reason") in _NONCE_REASONS and expected is not None:
                    nonce = int(expected)
                    continue
                self._check_accepted(response, request_id)
                with _SEQUENTIAL_GUARD:
                    _NEXT_SEQUENTIAL[key] = nonce + 1
                return request_id
            raise _rejection(response)
        finally:
            lock.release()

    # -- unresolved requests ---------------------------------------------

    def _remember(self, call_key: str, form: Dict[str, str], request_id: int) -> None:
        now = time.monotonic()
        self._unresolved[call_key] = _Pending(
            form, request_id, now + _UNRESOLVED_TTL_SECS
        )
        for key in [k for k, p in self._unresolved.items() if p.expires_at <= now]:
            del self._unresolved[key]
        while len(self._unresolved) > _UNRESOLVED_MAX:
            del self._unresolved[next(iter(self._unresolved))]

    def _forget(self, call_key: str) -> None:
        self._unresolved.pop(call_key, None)

    # -- polling ---------------------------------------------------------

    def _poll(self, request_id: int, deadline: float, **kwargs: Any) -> Dict[str, Any]:
        """Return the facilitator's final answer for ``request_id``.

        :raises MechRequestFailedError: for a delivered failure.
        :raises MechRequestRejectedError: when nothing was delivered.
        :raises MechDeadlineExceededError: when the answer is still pending,
            or the facilitator kept answering 5xx, until the budget ran out.
        """
        url = f"{self.offchain_url}/fetch_offchain_info"
        params = {"request_id": str(request_id)}
        while True:
            started = time.monotonic()
            try:
                response = self._http("GET", url, deadline, params=params, **kwargs)
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ):
                self._wait(deadline, self.poll_interval_secs)
                continue
            if response.status_code == 429:
                self._wait(deadline, _retry_after_secs(response))
                continue
            if response.status_code >= 500:
                # Says nothing about an accepted request; ask again.
                self._wait(deadline, self.poll_interval_secs)
                continue
            if response.status_code != 200:
                raise _rejection(response)
            body = _json_object(response)
            status = body.get("status")
            if status == "ok":
                return body
            if status == "failed":
                raise MechRequestFailedError(
                    request_id=request_id,
                    fault=str(body.get("fault") or FAULT_MECH),
                    error=str(body.get("error") or "request failed"),
                    delivery_rate=int(body.get("delivery_rate") or 0),
                )
            if status == "rejected":
                raise MechRequestRejectedError(
                    status_code=200,
                    error=str(body.get("fault") or "rejected"),
                    detail=str(body.get("reason") or "request was not delivered"),
                )
            # Still running; a held poll already waited, one answered at once did not.
            elapsed = time.monotonic() - started
            if elapsed < self.poll_interval_secs:
                self._wait(deadline, self.poll_interval_secs - elapsed)

    # -- the call --------------------------------------------------------

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        """Send the caller's upstream call as a standard mech request and return its result.

        :param request: the caller's prepared request to the upstream path.
        :param kwargs: adapter send arguments; a numeric ``timeout`` caps the
            whole call, which otherwise gets ``total_deadline_secs``.
        :return: a 200 response whose body is the tool's ``result``.
        """
        budget = self.total_deadline_secs
        timeout = kwargs.pop("timeout", None)
        if isinstance(timeout, (int, float)):
            budget = min(budget, float(timeout))
        deadline = time.monotonic() + budget
        prompt = self._prompt(request)
        pending = self._unresolved.get(prompt)
        if pending is not None and pending.expires_at > time.monotonic():
            request_id = self._resume(prompt, pending, deadline, **kwargs)
        else:
            request_id = None
        if request_id is None:
            ipfs_data = json.dumps(
                {"prompt": prompt, "tool": self.tool, "nonce": str(uuid.uuid4())}
            )
            request_id = self._submit(prompt, ipfs_data, deadline, **kwargs)
        try:
            answer = self._poll(request_id, deadline, **kwargs)
        except MechDeadlineExceededError as exc:
            raise MechOutcomeUnknownError(
                f"mech request {request_id} was accepted but its answer did not "
                f"arrive in time; the next identical call asks for it again"
            ) from exc
        except PaymentError:
            self._forget(prompt)
            raise
        self._forget(prompt)
        return self._as_response(request, answer, request_id)

    def _submit(
        self, call_key: str, ipfs_data: str, deadline: float, **kwargs: Any
    ) -> int:
        """Sign and post a fresh request; return its request id."""
        try:
            refusals = standard_explicit_nonce_refusals()
            if self.nonce_mode == NONCE_MODE_EXPLICIT and not refusals.refused(
                self.facilitator_base_url, self.chain
            ):
                accepted = self._submit_explicit(
                    call_key, ipfs_data, deadline, **kwargs
                )
                if accepted is not None:
                    return accepted
                _logger.warning(
                    "mech facilitator does not accept explicit nonces; resending "
                    "with the Safe's sequential nonce, and trying explicit again "
                    "in %.0f s",
                    refusals.retry_after_secs,
                )
                refusals.record(self.facilitator_base_url, self.chain)
            return self._submit_sequential(call_key, ipfs_data, deadline, **kwargs)
        except MechOutcomeUnknownError:
            # The form stays remembered: the facilitator may hold it.
            raise
        except PaymentError:
            self._forget(call_key)
            raise

    def _resume(
        self, call_key: str, pending: _Pending, deadline: float, **kwargs: Any
    ) -> Optional[int]:
        """Re-send a form whose outcome was unknown; its id, or ``None`` to sign afresh.

        A request the facilitator took is answered as already accepted, and
        one it never took is taken now, so neither is paid for twice.
        """
        _logger.info(
            "mech call has an unresolved request %s; asking about it first",
            pending.request_id,
        )
        response = self._post(pending.form, deadline, **kwargs)
        try:
            self._check_accepted(response, pending.request_id)
        except MechDepositRequiredError:
            raise
        except PaymentError:
            self._forget(call_key)
            return None
        return pending.request_id

    def _as_response(
        self,
        request: requests.PreparedRequest,
        answer: Dict[str, Any],
        request_id: int,
    ) -> requests.Response:
        """Return the tool's result as the upstream answer the caller expects."""
        delivered = answer.get("response")
        result = delivered.get("result") if isinstance(delivered, dict) else None
        if not isinstance(result, str):
            raise MechRequestRejectedError(
                status_code=200,
                error="no_result",
                detail=f"mech request {request_id} was delivered without a result",
            )
        response = requests.Response()
        response.status_code = 200
        response.reason = "OK"
        response._content = result.encode("utf-8")  # pylint: disable=protected-access
        response.encoding = "utf-8"
        response.headers["Content-Type"] = "application/json"
        response.headers["X-Mech-Request-Id"] = str(request_id)
        response.headers["X-Mech-Delivery-Rate"] = str(answer.get("delivery_rate", ""))
        response.url = str(request.url)
        response.request = request
        return response


def mech_standard_requests(  # pylint: disable=too-many-arguments
    account: Account,
    *,
    safe_address: str,
    chain: str,
    api: str,
    facilitator_base_url: str,
    delivery_rate_limit: Optional[int] = None,
    mech_address: Optional[str] = None,
    payment_type: Optional[bytes] = None,
    nonce_mode: NonceMode = NONCE_MODE_EXPLICIT,
    total_deadline_secs: float = DEFAULT_TOTAL_DEADLINE_SECS,
    default_timeout: Tuple[float, float] = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> requests.Session:
    """Create a session whose calls are standard off-chain requests to the facilitator mech.

    :param account: the agent EOA, sole owner of ``safe_address``.
    :param safe_address: the Safe that pays.
    :param chain: chain name, one of ``FACILITATOR_MECHS``.
    :param api: the tool, ``chat`` or ``coingecko``.
    :param facilitator_base_url: origin of the facilitator, no path.
    :param delivery_rate_limit: the most one request may be charged.
    :param mech_address: the facilitator mech, default for ``chain`` if None.
    :param payment_type: that mech's payment type, needed for a non-default mech.
    :param nonce_mode: ``explicit`` or ``sequential``.
    :param total_deadline_secs: wall-clock cap on one call, polls included.
    :param default_timeout: connect and read timeout of each HTTP call.
    :param kwargs: passed to ``HTTPAdapter``.
    :return: a session with the adapter mounted for http and https.
    """
    session = requests.Session()
    adapter = MechStandardHTTPAdapter(
        account,
        safe_address=safe_address,
        chain=chain,
        api=api,
        facilitator_base_url=facilitator_base_url,
        delivery_rate_limit=delivery_rate_limit,
        mech_address=mech_address,
        payment_type=payment_type,
        nonce_mode=nonce_mode,
        total_deadline_secs=total_deadline_secs,
        default_timeout=default_timeout,
        **kwargs,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
