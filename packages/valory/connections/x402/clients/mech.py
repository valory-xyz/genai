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
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
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
# Pause between retries of a refused slot. How long they go on for is
# ``nonce_retry_budget_secs`` below, not a count and not the call's deadline.
DEFAULT_NONCE_RETRY_WAIT_SECS = 2.0
# A refused slot clears when whatever holds the slot below it settles, and
# the whole of that wait is spent holding this Safe's lock. Waiting out the
# call's full deadline blocks every other route paying from the same Safe,
# so the retries get a shorter budget of their own and the call gives up.
DEFAULT_NONCE_RETRY_BUDGET_SECS = 60.0
# The facilitator may hold one POST for its admission wait (25s) plus RPC
# reads plus its upstream deadline (120s); an abandoned call is still charged.
FACILITATOR_WORST_CASE_SECS = 25.0 + 120.0
DEFAULT_MECH_TIMEOUT: Tuple[float, float] = (10.0, FACILITATOR_WORST_CASE_SECS + 20.0)
# How many times to wait out a 409 request_in_progress after a timeout,
# and a 409 requester_busy (another call from the same Safe in flight).
DEFAULT_IN_PROGRESS_RETRIES = 3
DEFAULT_BUSY_RETRIES = 3
DEFAULT_INFO_RATE_LIMIT_RETRIES = 3
# Wall-clock cap on one call, all waits, retries and sends included, so a
# struggling facilitator cannot hold the connection's single worker for
# many minutes: every send's read timeout is clamped to what is left.
DEFAULT_TOTAL_DEADLINE_SECS = 300.0
# A call that starts with less than this left signs, posts, and is cut
# off while the facilitator serves it, which is charged. Refuse instead.
# Must stay below ``total_deadline_secs`` or every call gives up.
DEFAULT_MIN_CALL_BUDGET_SECS = 30.0
# Replayable bodies kept per adapter, and how long one is kept before it
# is evicted to bound the store.
_UNRESOLVED_MAX = 32
_UNRESOLVED_TTL_SECS = 900.0

# One lock per (chain, Safe) for the whole process. The marketplace nonce
# belongs to the Safe, not to a session, and an agent holds one session per
# upstream api, so a per-session lock would let two of them race.
_SAFE_LOCKS: Dict[Tuple[str, str], threading.Lock] = {}
_SAFE_LOCKS_GUARD = threading.Lock()


def _safe_lock(chain: str, safe_address: str) -> threading.Lock:
    """Return the process-wide lock for one Safe on one chain.

    :param chain: facilitator chain slug.
    :param safe_address: the Safe that pays for the calls.
    :return: the lock shared by every adapter for that Safe.
    """
    key = (chain.lower(), safe_address.lower())
    with _SAFE_LOCKS_GUARD:
        lock = _SAFE_LOCKS.get(key)
        if lock is None:
            lock = _SAFE_LOCKS[key] = threading.Lock()
        return lock


class SlotRegistry:
    """Marketplace slots in use for a Safe, for one process.

    ``mapNonces`` only moves when a delivery settles, so a slot that is
    taken but unsettled is invisible to anyone reading the chain, and to
    any server that is not the one holding it. An agent can have more than
    one thing paying from a single Safe, and each of them tracks only its
    own, so one of these is the only complete view.

    Kept as a live object rather than exchanged as a value: a skill that
    signs marketplace requests without going through this adapter shares
    it by binding the same instance into the agent's shared state.
    """

    def __init__(self) -> None:
        """Start with no slots held for any Safe."""
        self._reserved: Dict[Tuple[str, str], Set[int]] = {}
        self._published: Dict[Tuple[str, str], Set[int]] = {}
        # When the signed request for a reserved slot stops being admissible,
        # on the facilitator's clock. Past that it can only be held by a
        # facilitator that already took it, which its own report settles.
        self._expiry: Dict[Tuple[Tuple[str, str], int], int] = {}
        self._guard = threading.Lock()

    def _forget(self, key: Tuple[str, str], slot: int) -> None:
        """Drop a reserved slot and its expiry; caller holds the guard."""
        reserved = self._reserved.get(key)
        if reserved is not None:
            reserved.discard(slot)
            if not reserved:
                del self._reserved[key]
        self._expiry.pop((key, slot), None)

    @property
    def live(self) -> Dict[Tuple[str, str], Set[int]]:
        """Return every slot in use per ``(chain, Safe)``, both lower case.

        :return: a snapshot, reserved here plus reported by the facilitator.

        A snapshot rather than the working set, because the two halves are
        retired differently: one when a caller hands a slot back, the other
        when the facilitator stops reporting it. Use ``clear`` to empty it.
        """
        with self._guard:
            keys = set(self._reserved) | set(self._published)
            merged = {
                key: self._reserved.get(key, set()) | self._published.get(key, set())
                for key in keys
            }
            return {key: slots for key, slots in merged.items() if slots}

    def clear(self) -> None:
        """Forget everything, for a process that wants a fresh start."""
        with self._guard:
            self._reserved.clear()
            self._published.clear()
            self._expiry.clear()

    def note_expiry(
        self, chain: str, safe_address: str, slot: int, expires_at: int
    ) -> None:
        """Record when the request signed at ``slot`` stops being admissible.

        :param chain: facilitator chain slug.
        :param safe_address: the Safe that pays for the call.
        :param slot: the slot that request was signed at.
        :param expires_at: unix seconds on the facilitator's clock.

        Held here rather than by the caller so any later read retires it.
        A caller that strands a slot and then goes idle would otherwise be
        the only thing able to free it, and everything else paying from the
        Safe would queue behind a slot nothing will ever settle.
        """
        key = (chain.lower(), safe_address.lower())
        with self._guard:
            self._expiry[(key, slot)] = int(expires_at)

    def publish(
        self,
        chain: str,
        safe_address: str,
        slots: Iterable[int],
        now: Optional[int] = None,
    ) -> None:
        """Replace what the facilitator is known to hold for ``safe_address``.

        :param chain: facilitator chain slug.
        :param safe_address: the Safe that pays for the call.
        :param slots: the slots it reports holding, from its requester info.
        :param now: the facilitator's clock, for retiring expired slots.

        Replaced wholesale rather than merged, because this is the only
        thing that can retire one of its rows. A row it gives up stops
        being reported, and nothing else in this process can tell: the
        chain counter never passes a slot that never settled.

        A slot it reports is its responsibility now, so it also stops
        being one of ours; keeping both would mean nothing ever released it.
        """
        key = (chain.lower(), safe_address.lower())
        moment = int(time.time() if now is None else now)
        with self._guard:
            held = {int(slot) for slot in slots}
            if held:
                self._published[key] = held
            else:
                self._published.pop(key, None)
            for slot in held & self._reserved.get(key, set()):
                # Its row now rather than ours.
                self._forget(key, slot)
            self._retire_expired_locked(key, moment)

    def _retire_expired_locked(self, key: Tuple[str, str], moment: int) -> List[int]:
        """Free reserved slots past their expiry; caller holds the guard.

        :param key: chain and Safe, both lower case.
        :param moment: the clock to judge expiry against.
        :return: the slots freed.

        Never touches what the facilitator reports. An expired request it
        already admitted is its row, and only its next report retires that.
        """
        freed = []
        for slot in list(self._reserved.get(key, set())):
            expires_at = self._expiry.get((key, slot))
            if expires_at is None or expires_at > moment:
                # No signed body yet, so the call that took it is still
                # inside its own attempt; or still admissible, so still ours.
                continue
            _logger.info(
                "mech slot %s expired at %s and no facilitator reports "
                "holding it; handing the slot back",
                slot,
                expires_at,
            )
            self._forget(key, slot)
            freed.append(slot)
        return freed

    def retire_expired(
        self, chain: str, safe_address: str, now: int
    ) -> List[int]:
        """Free slots whose signed request can no longer be admitted.

        :param chain: facilitator chain slug.
        :param safe_address: the Safe that pays for the call.
        :param now: the clock to judge expiry against.
        :return: the slots freed.

        For a caller with no facilitator of its own to ask. Slots the
        facilitator reports are left alone, so this is safe to run from
        anything in the process that can see a clock.
        """
        key = (chain.lower(), safe_address.lower())
        with self._guard:
            return self._retire_expired_locked(key, int(now))

    def hand_over(self, chain: str, safe_address: str, slot: int) -> None:
        """Record that the facilitator has taken responsibility for ``slot``.

        :param chain: facilitator chain slug.
        :param safe_address: the Safe that pays for the call.
        :param slot: the slot it acknowledged.

        Between the acknowledgement and the next requester-info read, the
        facilitator holds the slot but has not reported it yet. Without
        this the slot would belong to nobody for that window and something
        else paying from the Safe could sign it.
        """
        key = (chain.lower(), safe_address.lower())
        with self._guard:
            self._published.setdefault(key, set()).add(slot)
            self._forget(key, slot)

    def reserve(
        self, chain: str, safe_address: str, floor: int, settled_below: int
    ) -> int:
        """Take the lowest free slot at or above ``floor``.

        :param chain: facilitator chain slug.
        :param safe_address: the Safe that pays for the call.
        :param floor: lowest slot worth trying, from whichever server was
            asked. A server will refuse anything below its own answer.
        :param settled_below: the on-chain counter. Only slots below this
            have settled and can be forgotten.
        :return: the slot to sign at, now reserved.

        The two bounds are deliberately separate. A facilitator's first
        free slot sits above its own unsettled rows, so pruning at that
        number would forget slots it is still holding, and the next caller
        flooring at the on-chain counter would be handed one of them back.
        """
        key = (chain.lower(), safe_address.lower())
        with self._guard:
            reserved = self._reserved.setdefault(key, set())
            reserved.difference_update(
                [slot for slot in reserved if slot < settled_below]
            )
            in_use = reserved | self._published.get(key, set())
            slot = floor
            while slot in in_use:
                slot += 1
            reserved.add(slot)
            return slot

    def release(self, chain: str, safe_address: str, slot: int) -> None:
        """Give back a slot whose request was refused before it was served.

        :param chain: facilitator chain slug.
        :param safe_address: the Safe that pays for the call.
        :param slot: the slot reserved earlier.

        The marketplace consumes a requester's slots in order, so a slot
        kept after a refusal is one nothing will ever settle, and every
        later request for the Safe queues behind it forever.

        Only a slot reserved here. One the facilitator has reported is
        retired by it dropping out of a later report, not by this.
        """
        key = (chain.lower(), safe_address.lower())
        with self._guard:
            self._forget(key, slot)


_SLOTS = SlotRegistry()


def slot_registry() -> SlotRegistry:
    """Return the process-wide slot registry.

    :return: the registry every payer in this process shares.
    """
    return _SLOTS


def reserve_slot(
    chain: str, safe_address: str, floor: int, settled_below: int
) -> int:
    """Take the lowest free slot at or above ``floor``; see ``SlotRegistry``.

    :param chain: facilitator chain slug.
    :param safe_address: the Safe that pays for the call.
    :param floor: lowest slot worth trying.
    :param settled_below: the on-chain counter; only slots below it have settled.
    :return: the slot to sign at, now reserved.
    """
    return _SLOTS.reserve(chain, safe_address, floor, settled_below)


def release_slot(chain: str, safe_address: str, slot: int) -> None:
    """Hand a refused slot back; see ``SlotRegistry``.

    :param chain: facilitator chain slug.
    :param safe_address: the Safe that pays for the call.
    :param slot: the slot reserved earlier.
    """
    _SLOTS.release(chain, safe_address, slot)


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


class MechSlotHeldElsewhereError(MechDeadlineExceededError):
    """A slot below this call's stayed unsettled for the whole retry budget.

    Nothing was served or charged. Raised rather than waiting out the
    call's full deadline, because the wait holds this Safe's lock and
    every other route paying from it queues behind.
    """


class MechOutcomeUnknownError(PaymentError):
    """A signed request was sent and the replay that asks for its outcome failed too.

    The facilitator may have served and charged it; the same signed body
    is replayed first on the next call for the same upstream request.
    """


@dataclass(frozen=True)
class RequesterInfo:
    """Parsed ``GET /mech/{chain}/requester/{safe}`` response."""

    chain_id: int
    marketplace_address: str
    mech_address: str
    payment_type: bytes
    delivery_rates: Dict[str, int]
    next_nonce: int
    # ``mapNonces`` on the marketplace. Everything below it has settled;
    # ``next_nonce`` is higher whenever the facilitator holds rows of its
    # own, so the two are not interchangeable.
    on_chain_nonce: int
    balance: int
    available: int
    max_ttl_secs: int
    # The slots the facilitator holds. ``None`` from one that reports only
    # how many, where they cannot be derived: ``next_nonce`` is the first
    # slot with no live row, so a slot taken by another payer truncates the
    # run and hides every row above it.
    held_nonces: Optional[Tuple[int, ...]] = None
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
            on_chain_nonce=int(data["on_chain_nonce"]),
            balance=int(data["balance"]),
            available=int(data["available"]),
            max_ttl_secs=int(data["max_ttl_secs"]),
            held_nonces=(
                tuple(int(slot) for slot in data["held_nonces"])
                if data.get("held_nonces") is not None
                else None
            ),
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


def _clamp_timeout(
    timeout: Union[None, float, Tuple[float, float]], remaining: float
) -> Union[float, Tuple[float, float]]:
    """Return ``timeout`` with its read part capped at ``remaining`` seconds."""
    if timeout is None:
        return remaining
    if isinstance(timeout, tuple):
        connect, read = timeout
        return (connect, min(read, remaining))
    return min(timeout, remaining)


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
        nonce_retry_wait_secs: float = DEFAULT_NONCE_RETRY_WAIT_SECS,
        nonce_retry_budget_secs: float = DEFAULT_NONCE_RETRY_BUDGET_SECS,
        min_call_budget_secs: float = DEFAULT_MIN_CALL_BUDGET_SECS,
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
        :param nonce_retry_budget_secs: total time spent retrying a refused
            slot before giving up, so the per-Safe lock is not held for the
            whole deadline.
        :param nonce_retry_wait_secs: pause between nonce-collision retries,
            which continue until ``nonce_retry_budget_secs`` runs out.
        :param min_call_budget_secs: refuse to sign with less than this left.
        :param default_timeout: timeout applied when the caller passes none.
        :param total_deadline_secs: wall-clock cap on one call including retries.
        :param kwargs: passed to ``HTTPAdapter``.
        :raises ValueError: when the budget floor is not below the deadline.
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
        self.nonce_retry_wait_secs = nonce_retry_wait_secs
        self.nonce_retry_budget_secs = nonce_retry_budget_secs
        if min_call_budget_secs >= total_deadline_secs:
            raise ValueError(
                f"min_call_budget_secs ({min_call_budget_secs:.0f}s) must be "
                f"below total_deadline_secs ({total_deadline_secs:.0f}s), or "
                "every call is refused before it signs anything"
            )
        self.min_call_budget_secs = min_call_budget_secs
        self._default_timeout = default_timeout
        # Signed POSTs whose outcome is unknown, one entry per upstream call
        # so a second call does not discard the first one's replay.
        self._unresolved: "OrderedDict[str, Tuple[requests.PreparedRequest, int, float]]" = (
            OrderedDict()
        )
        # One call at a time per Safe; see ``_safe_lock``.
        self._lock = _safe_lock(self.chain, self.safe_address)
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

    def _check_budget(self, deadline: float) -> None:
        """Refuse a call that cannot finish, instead of paying to be cut off.

        :param deadline: monotonic instant after which the call gives up.
        :raises MechDeadlineExceededError: with too little budget left.

        The facilitator charges for a call it served even when the client
        walked away, so a call that starts with less than the minimum left
        is refused before it signs anything.
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MechDeadlineExceededError("mech call exceeded its budget")
        if remaining < self.min_call_budget_secs:
            raise MechDeadlineExceededError(
                f"mech call has {remaining:.1f}s left, below the "
                f"{self.min_call_budget_secs:.0f}s needed to see it through"
            )

    @staticmethod
    def _check(deadline: float) -> None:
        if time.monotonic() >= deadline:
            raise MechDeadlineExceededError("mech call exceeded its budget")

    def _send(
        self, prepared: requests.PreparedRequest, deadline: float, **kwargs: Any
    ) -> requests.Response:
        """Send with the read timeout clamped to what is left of ``deadline``."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise MechDeadlineExceededError("mech call exceeded its budget")
        kwargs = dict(kwargs)
        kwargs["timeout"] = _clamp_timeout(kwargs.get("timeout"), remaining)
        return super().send(prepared, **kwargs)

    def _fetch_info(self, deadline: float, **send_kwargs: Any) -> RequesterInfo:
        url = f"{self.facilitator_base_url}/mech/{self.chain}/requester/{self.safe_address}"
        prepared = requests.Request("GET", url).prepare()
        response = self._send(prepared, deadline, **send_kwargs)
        for _ in range(DEFAULT_INFO_RATE_LIMIT_RETRIES):
            if response.status_code != 429:
                break
            # The route is rate limited process-wide; it says when to retry.
            self._wait(deadline, _retry_after_secs(response))
            response = self._send(prepared, deadline, **send_kwargs)
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
        slot_registry().note_expiry(
            self.chain, self.safe_address, nonce, expires_at
        )
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
            response = self._send(prepared, deadline, **kwargs)
            if response.status_code not in _GATEWAY_STATUSES:
                return response
            _logger.warning(
                "mech request answered %s; asking for the stored response",
                response.status_code,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            _logger.warning("mech request failed; asking for the stored response")
        self._check(deadline)
        response = self._replay(prepared, deadline, **kwargs)
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
            response = self._replay(prepared, deadline, **kwargs)

    def _replay(
        self, prepared: requests.PreparedRequest, deadline: float, **kwargs: Any
    ) -> requests.Response:
        """Ask again for an already-sent signed body; a transport failure is a typed error."""
        try:
            return self._send(prepared, deadline, **kwargs)
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            raise MechOutcomeUnknownError(
                "mech request was sent but its outcome could not be read: "
                f"{type(exc).__name__}"
            ) from exc

    def _remember_unresolved(
        self, call_key: str, prepared: requests.PreparedRequest, nonce: int
    ) -> None:
        """Keep ``prepared`` replayable for ``call_key`` until it expires.

        :param call_key: canonical form of the upstream call being sent.
        :param prepared: the signed POST whose outcome is unknown.
        :param nonce: the slot it was signed at, so a later call can tell
            whether the facilitator ever admitted it.
        """
        now = time.monotonic()
        self._unresolved[call_key] = (prepared, nonce, now + _UNRESOLVED_TTL_SECS)
        self._unresolved.move_to_end(call_key)
        for key in [k for k, (_p, _n, e) in self._unresolved.items() if e <= now]:
            del self._unresolved[key]
        while len(self._unresolved) > _UNRESOLVED_MAX:
            self._unresolved.popitem(last=False)

    def _take_unresolved(
        self, call_key: str
    ) -> Optional[Tuple[requests.PreparedRequest, int]]:
        """Remove and return the replayable body for ``call_key`` and its slot.

        :param call_key: canonical form of the upstream call being sent.
        :return: the stored POST and its slot, or ``None`` when there is none.

        The stored expiry only drives eviction. Whether a body is still
        worth anything is the facilitator's call, and it answers a stale
        one with ``request_expired``, which signs afresh.
        """
        entry = self._unresolved.pop(call_key, None)
        return None if entry is None else (entry[0], entry[1])

    def _drop_stranded_slot(self, info: "RequesterInfo") -> None:
        """Free a slot held by a body the facilitator can no longer admit.

        :param info: the facilitator's requester info, read for this call.

        Only against a facilitator that reports a count alone. Where it
        reports the slots, the registry retires any expired slot it does
        not claim, which covers this and every slot above the first free
        one as well.

        A POST whose outcome never resolved keeps its slot, because the
        facilitator may still be serving it. If it was never admitted the
        facilitator's first free slot stays at that number, and once the
        signed request has expired it can never be admitted, so nothing
        will ever move the counter. Left alone that slot blocks every
        later call from this Safe until the process restarts.
        """
        if info.held_nonces is not None:
            return
        now = info.server_time if info.server_time is not None else int(time.time())
        for call_key, (prepared, nonce, _ttl) in list(self._unresolved.items()):
            if nonce != int(info.next_nonce):
                continue
            try:
                expires_at = int(json.loads(prepared.body or b"{}")["expires_at"])
            except (ValueError, KeyError, TypeError):
                continue
            if expires_at > now:
                continue
            _logger.info(
                "mech slot %s is held by a request that expired at %s and the "
                "facilitator never admitted it; handing the slot back",
                nonce,
                expires_at,
            )
            del self._unresolved[call_key]
            self._release(nonce)

    def _resume_unresolved(
        self, call_key: str, info: "RequesterInfo", deadline: float, **kwargs: Any
    ) -> Optional[requests.Response]:
        """Replay the last unresolved signed body if it was for this same upstream call.

        :param call_key: canonical form of the upstream call being sent.
        :param info: the facilitator's requester info, read for this call.
        :param deadline: monotonic instant after which no more waiting is done.
        :param kwargs: adapter send arguments.
        :return: the stored response, or ``None`` when a fresh request must be signed.

        Replayed unless the body's slot is above the facilitator's first
        free slot. At it the body may still be queued for admission, and
        signing afresh would have the original admitted behind the retry
        and pay for both; below it the slot is already spent, so the body
        was admitted and its outcome is the thing worth asking for. Above
        it the facilitator has not reached that slot and never will while
        the gap stands, so the body is dead and a fresh one is signed.
        """
        pending = self._take_unresolved(call_key)
        if pending is None:
            return None
        prepared, nonce = pending
        if nonce > int(info.next_nonce):
            _logger.info(
                "mech call has an unresolved request at slot %s, above the "
                "facilitator's first free slot, which it will never reach; "
                "signing afresh",
                nonce,
            )
            # The marketplace consumes a requester's slots in order, so
            # nothing will ever settle this one and leaving it reserved
            # would stall every later request for this Safe.
            self._release(nonce)
            return None
        _logger.info(
            "mech call has an unresolved request; asking for its outcome first"
        )
        try:
            response = self._post_signed(prepared, deadline, **kwargs)
        except BaseException:
            # The outcome is no clearer than it was, so the body has to stay
            # replayable; losing it here is what makes the next call pay twice.
            self._remember_unresolved(call_key, prepared, nonce)
            raise
        if 200 <= response.status_code < 300:
            self._hand_over(nonce, info)
            return response
        parsed = _facilitator_error(response)
        if parsed is None:
            # An upstream answer passed through: the call was served and
            # charged, so it is the caller's to handle, not one to re-sign.
            self._hand_over(nonce, info)
            return response
        if parsed[0] == _IN_PROGRESS:
            # Still being served, so it is its row; kept replayable too.
            self._hand_over(nonce, info)
            self._remember_unresolved(call_key, prepared, nonce)
            raise MechRequestRejectedError(
                status_code=response.status_code, error=parsed[0], detail=parsed[1]
            )
        # Refused: it was never served, so nothing holds the slot. The
        # marketplace consumes a requester's slots in order, so a slot left
        # reserved with nothing to settle it stalls every later call.
        self._release(nonce)
        return None

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:  # type: ignore[override]
        """Sign the call for the marketplace and post it to the facilitator.

        :param request: the caller's prepared request to the upstream path.
        :param kwargs: adapter send arguments; ``timeout`` defaults when absent.
        :return: the upstream response as returned by the facilitator.
        """
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self._default_timeout
        # Started before the lock, so time spent waiting for another thread
        # counts against this call's budget instead of extending it.
        deadline = time.monotonic() + self.total_deadline_secs
        if not self._lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise MechDeadlineExceededError(
                "mech call gave up waiting for another request on this Safe"
            )
        try:
            self._check_budget(deadline)
            return self._send_locked(request, deadline, **kwargs)
        finally:
            self._lock.release()

    def _reserve(self, info: "RequesterInfo") -> int:
        """Take a slot for this Safe, at or above the facilitator's answer.

        :param info: the facilitator's requester info.
        :return: the slot to sign at.

        The facilitator's answer is the on-chain counter plus its own
        unsettled rows, so it is a floor and not the whole picture: it
        cannot see slots held by anything else paying from this Safe.
        """
        return reserve_slot(
            self.chain,
            self.safe_address,
            int(info.next_nonce),
            int(info.on_chain_nonce),
        )

    def _release(self, nonce: int) -> None:
        """Hand back a slot the facilitator refused before serving it.

        :param nonce: the slot reserved earlier.
        """
        release_slot(self.chain, self.safe_address, nonce)

    def _publish(self, info: "RequesterInfo") -> None:
        """Record what the facilitator says it holds for this Safe.

        :param info: the facilitator's requester info.

        Skipped against a facilitator that reports only a count, where the
        set cannot be worked out, and the older release-by-rule behaviour
        stands instead.
        """
        if info.held_nonces is None:
            return
        slot_registry().publish(
            self.chain,
            self.safe_address,
            info.held_nonces,
            now=info.server_time,
        )

    def _hand_over(self, nonce: int, info: "RequesterInfo") -> None:
        """Give a slot the facilitator has acknowledged over to it.

        :param nonce: the slot it accepted.
        :param info: the facilitator's requester info for this call.

        Only against a facilitator that reports the slots it holds, since
        that report is the only thing that retires one. Handing a slot to
        one that reports a count alone would leave it held for the life of
        the process, which is worse than keeping it ourselves: at least the
        chain counter prunes ours once the delivery settles.
        """
        if info.held_nonces is None:
            return
        slot_registry().hand_over(self.chain, self.safe_address, nonce)

    def _send_locked(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        self, request: requests.PreparedRequest, deadline: float, **kwargs: Any
    ) -> requests.Response:
        """Run one signed call; the caller holds the per-Safe lock.

        :param request: the caller's prepared request to the upstream path.
        :param deadline: monotonic instant after which the call gives up.
        :param kwargs: adapter send arguments.
        :return: the upstream response as returned by the facilitator.
        """
        call = self._upstream_call(request)
        call_key = json.dumps(call, sort_keys=True)
        # Read first: whether a stored body is worth replaying depends on
        # where the facilitator's first free slot sits relative to it.
        info = self._fetch_info(deadline, **kwargs)
        self._publish(info)
        self._drop_stranded_slot(info)
        resumed = self._resume_unresolved(call_key, info, deadline, **kwargs)
        if resumed is not None:
            return resumed
        nonce = self._reserve(info)
        url = f"{self.facilitator_base_url}/mech/{self.api}/{self.chain}"
        busy_budget: Optional[float] = None
        # Whether the facilitator may be holding the slot. Handing back one
        # it is still serving would let something else in this agent sign
        # the same slot, which is the collision this registry exists to
        # stop. So anything short of a definite refusal keeps it.
        consumed = False

        attempt = 0
        refused: Optional[Tuple[int, Any, int]] = None
        nonce_deadline: Optional[float] = None
        try:
            while True:
                body = self._signed_body(call, info, nonce)
                prepared = requests.Request(
                    "POST",
                    url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(body),
                ).prepare()
                # Before every signed POST, not just the first: a retry that
                # starts with seconds left is served and charged the same way.
                self._check_budget(deadline)
                # Remembered until a definite outcome: if the deadline or the
                # transport fails after this POST, the next call for the same
                # upstream request replays it instead of paying twice.
                self._remember_unresolved(call_key, prepared, nonce)
                consumed = True
                response = self._post_signed(prepared, deadline, **kwargs)
                if 200 <= response.status_code < 300:
                    self._take_unresolved(call_key)
                    # Its row from here. Kept as ours it would never be
                    # released: the chain counter cannot pass a slot the
                    # facilitator later gives up on, so only its own next
                    # report can retire it.
                    self._hand_over(nonce, info)
                    return response

                parsed = _facilitator_error(response)
                if parsed is None:
                    self._take_unresolved(call_key)
                    self._hand_over(nonce, info)
                    return response
                error, detail, context = parsed
                if error == _IN_PROGRESS:
                    # Still being served at that slot, so it is its row.
                    self._hand_over(nonce, info)
                else:
                    self._take_unresolved(call_key)
                    consumed = False
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
                        self._publish(info)
                        self._release(nonce)
                        nonce = self._reserve(info)
                        continue
                if response.status_code == 409 and error == _NONCE_MISMATCH:
                    # The slot clears when the request ahead settles, which is
                    # a settlement tick away, so this waits on time rather than
                    # a count, bounded by ``nonce_retry_budget_secs``.
                    attempt += 1
                    refused = (nonce, context.get("expected"), attempt)
                    _logger.info(
                        "mech slot %s refused (facilitator expects %s), attempt %s",
                        *refused,
                    )
                    if nonce_deadline is None:
                        nonce_deadline = min(
                            deadline,
                            time.monotonic() + self.nonce_retry_budget_secs,
                        )
                    if time.monotonic() + self.nonce_retry_wait_secs > nonce_deadline:
                        slot, expected, attempts = refused
                        raise MechSlotHeldElsewhereError(
                            f"mech slot {slot} was still refused after {attempts} "
                            f"attempts over {self.nonce_retry_budget_secs:.0f}s; "
                            f"the facilitator expected {expected}"
                        )
                    self._wait(deadline, self.nonce_retry_wait_secs)
                    info = self._fetch_info(deadline, **kwargs)
                    self._publish(info)
                    self._release(nonce)
                    nonce = self._reserve(info)
                    continue
                raise MechRequestRejectedError(
                    status_code=response.status_code, error=error, detail=detail
                )
        except MechSlotHeldElsewhereError:
            raise
        except MechDeadlineExceededError as exc:
            if refused is None:
                raise
            # Without this the error only says the budget ran out, which
            # reads like a slow upstream rather than a slot held elsewhere.
            slot, expected, attempts = refused
            raise MechDeadlineExceededError(
                f"{exc} after {attempts} attempts at slot {slot}; the "
                f"facilitator expected {expected}"
            ) from exc
        finally:
            if not consumed:
                self._release(nonce)


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
    total_deadline_secs: float = DEFAULT_TOTAL_DEADLINE_SECS,
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
    :param total_deadline_secs: wall-clock cap on one call, everything included.
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
        total_deadline_secs=total_deadline_secs,
        **kwargs,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
