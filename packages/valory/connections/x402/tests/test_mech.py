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

"""Tests for the mech-marketplace payment adapter and its signing primitives."""

# pylint: disable=protected-access,redefined-outer-name

import base64
import json
import time
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

import pytest
import requests
from eth_account import Account
from eth_utils import keccak, to_checksum_address

from packages.valory.connections.x402.clients import mech as mech_module
from packages.valory.connections.x402.clients.base import PaymentError
from packages.valory.connections.x402.clients.mech import (
    DEFAULT_MECH_TIMEOUT,
    DEFAULT_REQUEST_TTL_SECS,
    MechDepositRequiredError,
    MechRateExceededError,
    MechRequestRejectedError,
    RequesterInfo,
    mech_requests,
)
from packages.valory.connections.x402.mech_signing import (
    canonical_request_data,
    compute_domain_separator,
    compute_safe_message_hash,
    derive_request_id,
)

# ---------------------------------------------------------------------------
# Signing primitives: pinned against the mech_interact golden vector and the
# facilitator's canonical-bytes vector, so a drift on either side fails here.
# ---------------------------------------------------------------------------

_MARKETPLACE = "0x" + "11" * 20
_MECH = "0x" + "22" * 20
_REQUESTER = "0x" + "33" * 20
_DATA = b'{"prompt":"hi","tool":"t","nonce":"n"}'
_DELIVERY_RATE = 10**16
_PAYMENT_TYPE = bytes.fromhex(
    "ba699a34be8fe0e7725e93dcbce1701b0211a8ca61330aaeb8a05bf2ec7abed1"
)
_NONCE = 42
_CHAIN_ID = 100
_EXPECTED_DOMAIN = "df50cdbe42bf9d9976fcb9374107c6cf0450b566360eaa04dacb3bb0c1cc8845"
_EXPECTED_REQUEST_ID = (
    "76a226b6cf8c7da71bd6340d00e54002d9b01d1e8db5670d4bd3282863633fb0"
)
_SAFE_FOR_VECTOR = "0x" + "44" * 20
_EXPECTED_SAFE_MESSAGE = (
    "1f045c3aa1ba9e74c2ff21d51cc0be889abe3e7f8d1e1ffaaafa16ab635d9855"
)


def _golden_request_id(**overrides: Any) -> bytes:
    kwargs: Dict[str, Any] = {
        "marketplace_address": _MARKETPLACE,
        "mech_address": _MECH,
        "requester": _REQUESTER,
        "data": _DATA,
        "delivery_rate": _DELIVERY_RATE,
        "payment_type": _PAYMENT_TYPE,
        "nonce": _NONCE,
        "chain_id": _CHAIN_ID,
    }
    kwargs.update(overrides)
    return derive_request_id(**kwargs)


def test_domain_separator_matches_mech_interact_vector() -> None:
    """The asymmetric name/version hashing matches the contract byte for byte."""
    assert compute_domain_separator(_CHAIN_ID, _MARKETPLACE).hex() == _EXPECTED_DOMAIN


def test_request_id_matches_mech_interact_vector() -> None:
    """Same inputs as the mech_interact skill give the same request_id."""
    assert _golden_request_id().hex() == _EXPECTED_REQUEST_ID


@pytest.mark.parametrize(
    "override",
    [
        {"nonce": _NONCE + 1},
        {"data": _DATA + b" "},
        {"delivery_rate": _DELIVERY_RATE + 1},
        {"chain_id": _CHAIN_ID + 1},
        {"requester": "0x" + "34" * 20},
    ],
)
def test_every_request_id_input_is_committed(override: Dict[str, Any]) -> None:
    """Changing any single input changes the request_id."""
    assert _golden_request_id(**override).hex() != _EXPECTED_REQUEST_ID


def test_request_id_rejects_wrong_payment_type_length() -> None:
    """A payment type that is not 32 bytes is refused up front."""
    with pytest.raises(ValueError):
        _golden_request_id(payment_type=b"\x00")


def test_safe_message_hash_matches_mech_interact() -> None:
    """The Safe wrap equals ``compute_safe_message_hash`` in the mech_interact skill."""
    digest = compute_safe_message_hash(
        bytes.fromhex(_EXPECTED_REQUEST_ID), _SAFE_FOR_VECTOR, _CHAIN_ID
    )
    assert digest.hex() == _EXPECTED_SAFE_MESSAGE


def test_safe_message_hash_rejects_wrong_length() -> None:
    """Only a 32-byte request_id can be wrapped."""
    with pytest.raises(ValueError):
        compute_safe_message_hash(b"\x01", _SAFE_FOR_VECTOR, _CHAIN_ID)


def test_canonical_bytes_match_the_facilitator_vector() -> None:
    """The facilitator pins these exact bytes; the client must produce them."""
    expected = (
        '{"body_b64":"eyJwcm9tcHQiOiJoaSJ9","expires_at":1700000060,'
        '"method":"POST","path":"/models/gemini-2.5-flash:generateContent",'
        '"query":{"a":"1","b":"2"}}'
    ).encode("utf-8")
    actual = canonical_request_data(
        method="post",
        path="/models/gemini-2.5-flash:generateContent",
        query={"b": "2", "a": "1"},
        body_b64="eyJwcm9tcHQiOiJoaSJ9",
        expires_at=1_700_000_060,
    )
    assert actual == expected


# ---------------------------------------------------------------------------
# Adapter: driven through a real ``requests.Session`` with ``HTTPAdapter.send``
# replaced by a scripted fake facilitator.
# ---------------------------------------------------------------------------

_FACILITATOR = "https://facilitator.example"
_CHAIN = "optimism"
_API = "chat"
_SAFE = to_checksum_address("0x0000000000000000000000000000000000000111")
_INFO_JSON: Dict[str, Any] = {
    "chain": _CHAIN,
    "chain_id": 10,
    "marketplace_address": "0x000000000000000000000000000000000000dEaD",
    "mech_address": "0x000000000000000000000000000000000000BeEF",
    "payment_type": "0x6406bb5f31a732f898e1ce9fdd988a80a808d36ab5d9a4a4805a8be8d197d5e3",
    "delivery_rates": {"chat": 10000, "coingecko": 5000},
    "next_nonce": 7,
    "on_chain_nonce": 7,
    "held": 0,
    "balance": 50000,
    "available": 50000,
    "max_ttl_secs": 300,
}
_UPSTREAM_BODY = b'{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}'


def _response(status: int, body: bytes, headers: Optional[Dict[str, str]] = None):
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.headers.update(headers or {"content-type": "application/json"})
    return response


def _json_response(status: int, payload: Any) -> requests.Response:
    return _response(status, json.dumps(payload).encode("utf-8"))


class _FakeFacilitator:
    """Scripted responses per (method, path), recording every call."""

    def __init__(self, post_responses: List[Any]) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._posts = list(post_responses)
        self.info = dict(_INFO_JSON)
        self.info_headers: Dict[str, str] = {"content-type": "application/json"}
        # Replaces ``info`` after the first POST, to model another call landing.
        self.info_after_post: Optional[Dict[str, Any]] = None

    def send(self, _adapter: Any, request: requests.PreparedRequest, **kwargs: Any):
        self.calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "body": request.body,
                "timeout": kwargs.get("timeout"),
            }
        )
        if request.method == "GET":
            return _response(
                200, json.dumps(self.info).encode("utf-8"), self.info_headers
            )
        assert self._posts, "unexpected POST"
        item = self._posts.pop(0)
        if self.info_after_post is not None:
            self.info = self.info_after_post
        if isinstance(item, Exception):
            raise item
        return item


def _session_with(
    fake: _FakeFacilitator,  # pylint: disable=unused-argument
    account: Optional[Account] = None,
    api: str = _API,
    chain: str = _CHAIN,
    **kwargs: Any,
) -> Callable[[], requests.Session]:
    def _make() -> requests.Session:
        return mech_requests(
            account or Account.create(),
            safe_address=_SAFE,
            chain=chain,
            api=api,
            facilitator_base_url=_FACILITATOR,
            **kwargs,
        )

    return _make


def _patched(fake: _FakeFacilitator):
    return patch.object(
        requests.adapters.HTTPAdapter, "send", autospec=True, side_effect=fake.send
    )


def _posted_body(fake: _FakeFacilitator, index: int = 0) -> Dict[str, Any]:
    posts = [c for c in fake.calls if c["method"] == "POST"]
    return json.loads(posts[index]["body"])


def test_happy_path_posts_a_signed_body_and_returns_the_upstream_response() -> None:
    """One info GET, one signed POST, and the facilitator's body comes back as-is."""
    account = Account.create()
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    session = _session_with(fake, account)()

    with _patched(fake):
        response = session.post(
            f"{_FACILITATOR}/v1beta/models/gemini-2.5-flash:generateContent?alt=json",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"contents": []}),
        )

    assert response.status_code == 200
    assert response.content == _UPSTREAM_BODY
    assert [c["method"] for c in fake.calls] == ["GET", "POST"]
    assert fake.calls[0]["url"] == f"{_FACILITATOR}/mech/{_CHAIN}/requester/{_SAFE}"
    assert fake.calls[1]["url"] == f"{_FACILITATOR}/mech/{_API}/{_CHAIN}"

    body = _posted_body(fake)
    assert body["safe_address"] == _SAFE
    assert body["nonce"] == "7"
    assert body["upstream"] == {
        "method": "POST",
        "path": "/v1beta/models/gemini-2.5-flash:generateContent",
        "query": {"alt": "json"},
        "body_b64": base64.b64encode(json.dumps({"contents": []}).encode()).decode(),
    }
    # The facilitator re-derives the request_id from the body; do the same here.
    data = canonical_request_data(
        method="POST",
        path=body["upstream"]["path"],
        query=body["upstream"]["query"],
        body_b64=body["upstream"]["body_b64"],
        expires_at=body["expires_at"],
    )
    request_id = derive_request_id(
        marketplace_address=_INFO_JSON["marketplace_address"],
        mech_address=_INFO_JSON["mech_address"],
        requester=_SAFE,
        data=data,
        delivery_rate=_INFO_JSON["delivery_rates"]["chat"],
        payment_type=bytes.fromhex(_INFO_JSON["payment_type"][2:]),
        nonce=7,
        chain_id=_INFO_JSON["chain_id"],
    )
    assert body["request_id"] == "0x" + request_id.hex()
    # The signature recovers to the EOA over the Safe-wrapped digest.
    digest = compute_safe_message_hash(request_id, _SAFE, _INFO_JSON["chain_id"])
    recovered = Account._recover_hash(  # pylint: disable=no-value-for-parameter
        digest, signature=bytes.fromhex(body["signature"][2:])
    )
    assert recovered == account.address
    assert int(body["signature"][-2:], 16) in (27, 28)


def test_expires_at_is_bounded_by_the_facilitator_max_ttl() -> None:
    """The client asks for its TTL but never beyond what the facilitator accepts."""
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    fake.info["max_ttl_secs"] = 30
    session = _session_with(fake, ttl_secs=DEFAULT_REQUEST_TTL_SECS)()

    with patch(
        "packages.valory.connections.x402.clients.mech.time.time", lambda: 1000.0
    ):
        with _patched(fake):
            session.get(f"{_FACILITATOR}/api/v3/simple/price")

    assert _posted_body(fake)["expires_at"] == 1030


def test_get_request_has_empty_body_and_carries_the_query() -> None:
    """A GET forwards its query and an empty body_b64."""
    fake = _FakeFacilitator([_response(200, b"{}")])
    session = _session_with(fake, api="coingecko")()

    with _patched(fake):
        session.get(
            f"{_FACILITATOR}/api/v3/simple/price?ids=ethereum&vs_currencies=usd"
        )

    upstream = _posted_body(fake)["upstream"]
    assert upstream["method"] == "GET"
    assert upstream["body_b64"] == ""
    assert upstream["query"] == {"ids": "ethereum", "vs_currencies": "usd"}
    assert fake.calls[1]["url"] == f"{_FACILITATOR}/mech/coingecko/{_CHAIN}"


def test_nonce_collision_retries_at_the_reported_slot() -> None:
    """A 409 nonce_mismatch is retried once at the slot the facilitator names."""
    collision = _json_response(
        409,
        {
            "detail": {
                "error": "nonce_mismatch",
                "detail": "next slot",
                "context": {"expected": "9", "got": "7"},
            }
        },
    )
    fake = _FakeFacilitator([collision, _response(200, _UPSTREAM_BODY)])
    session = _session_with(fake)()

    with _patched(fake):
        response = session.get(f"{_FACILITATOR}/x")

    assert response.status_code == 200
    assert _posted_body(fake, 0)["nonce"] == "7"
    assert _posted_body(fake, 1)["nonce"] == "9"
    # The signature was rebuilt for the new nonce, not reused.
    assert _posted_body(fake, 0)["request_id"] != _posted_body(fake, 1)["request_id"]


def test_nonce_retries_are_bounded() -> None:
    """After the configured retries a persistent 409 surfaces as a rejection."""
    collision = _json_response(
        409,
        {
            "detail": {
                "error": "nonce_mismatch",
                "detail": "next slot",
                "context": {"expected": "8", "got": "7"},
            }
        },
    )
    fake = _FakeFacilitator([collision, collision, collision])
    session = _session_with(fake, nonce_retries=2)()

    with _patched(fake), pytest.raises(MechRequestRejectedError) as excinfo:
        session.get(f"{_FACILITATOR}/x")

    assert excinfo.value.status_code == 409
    assert excinfo.value.error == "nonce_mismatch"
    assert len([c for c in fake.calls if c["method"] == "POST"]) == 3


def test_402_raises_a_typed_deposit_error_with_the_shortfall() -> None:
    """The deposit trigger needs the numbers, not just an error string."""
    short = _json_response(
        402,
        {
            "detail": {
                "error": "insufficient_pre_deposit",
                "detail": "top up",
                "context": {
                    "balance": "12000",
                    "reserved": "10000",
                    "available": "2000",
                    "required": "10000",
                },
            }
        },
    )
    fake = _FakeFacilitator([short])
    session = _session_with(fake)()

    with _patched(fake), pytest.raises(MechDepositRequiredError) as excinfo:
        session.get(f"{_FACILITATOR}/x")

    err = excinfo.value
    assert (err.balance, err.reserved, err.available, err.required) == (
        12000,
        10000,
        2000,
        10000,
    )
    assert isinstance(err, PaymentError)


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (401, {"detail": {"error": "invalid_signature", "detail": "bad sig"}}),
        (503, {"detail": {"error": "settlement_paused", "detail": "held"}}),
        (400, {"detail": "request_id does not match the derived value for this body"}),
    ],
)
def test_facilitator_errors_raise_a_typed_rejection(status: int, body: Any) -> None:
    """Facilitator-side refusals never come back as a plain response."""
    fake = _FakeFacilitator([_json_response(status, body)])
    session = _session_with(fake)()

    with _patched(fake), pytest.raises(MechRequestRejectedError) as excinfo:
        session.get(f"{_FACILITATOR}/x")

    assert excinfo.value.status_code == status


def test_upstream_error_without_facilitator_shape_is_passed_through() -> None:
    """A CoinGecko 429 or Gemini 400 reaches the caller unchanged, as it does today."""
    upstream_429 = _json_response(429, {"status": {"error_code": 429}})
    fake = _FakeFacilitator([upstream_429])
    session = _session_with(fake)()

    with _patched(fake):
        response = session.get(f"{_FACILITATOR}/x")

    assert response.status_code == 429
    assert response.json() == {"status": {"error_code": 429}}


def test_rate_above_the_cap_is_refused_before_signing() -> None:
    """The client never signs for more than it was configured to accept."""
    fake = _FakeFacilitator([])
    session = _session_with(fake, max_delivery_rate=9999)()

    with _patched(fake), pytest.raises(MechRateExceededError):
        session.get(f"{_FACILITATOR}/x")

    assert [c["method"] for c in fake.calls] == ["GET"]


def test_info_fetch_failure_is_a_rejection() -> None:
    """If the requester info cannot be read nothing is signed or sent."""
    fake = _FakeFacilitator([])
    fake.send = lambda _a, request, **_k: _json_response(  # type: ignore[assignment]
        404, {"detail": {"error": "unknown_chain", "detail": "no such chain"}}
    )
    session = _session_with(fake)()

    with _patched(fake), pytest.raises(MechRequestRejectedError) as excinfo:
        session.get(f"{_FACILITATOR}/x")

    assert excinfo.value.error == "unknown_chain"


def test_request_to_another_host_is_refused() -> None:
    """Only the configured facilitator origin is ever signed for."""
    fake = _FakeFacilitator([])
    session = _session_with(fake)()

    with _patched(fake), pytest.raises(PaymentError):
        session.get("https://elsewhere.example/x")

    assert fake.calls == []


def test_default_timeout_is_injected_when_caller_omits() -> None:
    """Both the info GET and the signed POST carry the adapter default timeout."""
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    session = _session_with(fake)()

    with _patched(fake):
        session.get(f"{_FACILITATOR}/x")

    assert [c["timeout"] for c in fake.calls] == [DEFAULT_MECH_TIMEOUT] * 2


def test_explicit_timeout_is_preserved() -> None:
    """A caller-supplied timeout flows through unchanged."""
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    session = _session_with(fake)()

    with _patched(fake):
        session.get(f"{_FACILITATOR}/x", timeout=3)

    assert [c["timeout"] for c in fake.calls] == [3, 3]


def test_requester_info_parses_hex_payment_type_and_ints() -> None:
    """Numbers arrive as JSON ints or strings; the parsed form is typed."""
    info = RequesterInfo.from_json({**_INFO_JSON, "next_nonce": "12", "balance": "5"})

    assert info.next_nonce == 12
    assert info.balance == 5
    assert info.payment_type == bytes.fromhex(_INFO_JSON["payment_type"][2:])
    assert len(info.payment_type) == 32
    assert keccak(text="FixedPriceTokenUSDC") == info.payment_type


def test_read_timeout_asks_once_more_for_the_same_request_id() -> None:
    """A call the client stopped waiting for is collected, not paid for twice."""
    fake = _FakeFacilitator(
        [requests.exceptions.ReadTimeout("slow"), _response(200, _UPSTREAM_BODY)]
    )
    session = _session_with(fake)()

    with _patched(fake):
        response = session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert response.status_code == 200
    assert _posted_body(fake, 0)["request_id"] == _posted_body(fake, 1)["request_id"]
    assert [c["method"] for c in fake.calls] == ["GET", "POST", "POST"]


def test_second_read_timeout_is_raised() -> None:
    fake = _FakeFacilitator(
        [
            requests.exceptions.ReadTimeout("slow"),
            requests.exceptions.ReadTimeout("slow"),
        ]
    )
    session = _session_with(fake)()

    with _patched(fake), pytest.raises(requests.exceptions.ReadTimeout):
        session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})


def test_default_timeout_outlasts_the_facilitator_upstream_deadline() -> None:
    """The facilitator serves and charges a call for up to its 120s deadline plus RPC reads."""
    assert DEFAULT_MECH_TIMEOUT[1] > 120


def test_expires_at_follows_the_facilitator_clock() -> None:
    """The Date header on the info response, not the agent clock, anchors expires_at."""
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    fake.info_headers["Date"] = "Wed, 01 Jan 2031 00:00:00 GMT"
    server_now = 1_924_992_000  # the header above, in unix seconds
    session = _session_with(fake, ttl_secs=60)()

    with _patched(fake):
        session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert _posted_body(fake)["expires_at"] == server_now + 60


def test_malformed_date_header_falls_back_to_the_local_clock() -> None:
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    fake.info_headers["Date"] = "not a date"
    session = _session_with(fake, ttl_secs=60)()
    before = int(time.time())

    with _patched(fake):
        session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert before + 60 <= _posted_body(fake)["expires_at"] <= int(time.time()) + 60


def test_chain_slug_is_lowercased_on_the_wire() -> None:
    fake = _FakeFacilitator([_response(200, _UPSTREAM_BODY)])
    session = _session_with(fake, chain="Optimism")()

    with _patched(fake):
        session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert fake.calls[0]["url"].startswith(f"{_FACILITATOR}/mech/optimism/requester/")
    assert fake.calls[1]["url"] == f"{_FACILITATOR}/mech/{_API}/optimism"


def test_in_progress_after_a_timeout_is_waited_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry may find the original still finishing; Retry-After is honoured until it lands."""
    in_progress = _json_response(
        409,
        {"detail": {"error": "request_in_progress", "detail": "busy", "context": {}}},
    )
    in_progress.headers["Retry-After"] = "1"
    fake = _FakeFacilitator(
        [
            requests.exceptions.ReadTimeout("slow"),
            in_progress,
            in_progress,
            _response(200, _UPSTREAM_BODY),
        ]
    )
    session = _session_with(fake)()
    slept: List[float] = []
    monkeypatch.setattr(mech_module.time, "sleep", slept.append)

    with _patched(fake):
        response = session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert response.status_code == 200
    assert slept == [1.0, 1.0]
    request_ids = {_posted_body(fake, i)["request_id"] for i in range(4)}
    assert len(request_ids) == 1


def test_in_progress_that_never_clears_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_progress = _json_response(
        409,
        {"detail": {"error": "request_in_progress", "detail": "busy", "context": {}}},
    )
    fake = _FakeFacilitator(
        [requests.exceptions.ReadTimeout("slow")] + [in_progress] * 4
    )
    session = _session_with(fake)()
    monkeypatch.setattr(mech_module.time, "sleep", lambda _s: None)

    with _patched(fake), pytest.raises(MechRequestRejectedError) as excinfo:
        session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert excinfo.value.error == "request_in_progress"


def _busy_response() -> requests.Response:
    response = _json_response(
        409, {"detail": {"error": "requester_busy", "detail": "busy", "context": {}}}
    )
    response.headers["Retry-After"] = "1"
    return response


def test_requester_busy_is_waited_out_and_the_nonce_re_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another call from the same Safe in flight: wait, then sign at the fresh next nonce."""
    fake = _FakeFacilitator([_busy_response(), _response(200, _UPSTREAM_BODY)])
    fake.info_after_post = dict(_INFO_JSON, next_nonce=8)  # the other call lands
    session = _session_with(fake)()
    slept: List[float] = []
    monkeypatch.setattr(mech_module.time, "sleep", slept.append)

    with _patched(fake):
        response = session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert response.status_code == 200
    assert slept == [1.0]
    assert [c["method"] for c in fake.calls] == ["GET", "POST", "GET", "POST"]
    assert _posted_body(fake, 0)["nonce"] == "7"
    assert _posted_body(fake, 1)["nonce"] == "8"


def test_requester_busy_that_never_clears_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeFacilitator([_busy_response()] * 5)
    session = _session_with(fake)()
    monkeypatch.setattr(mech_module.time, "sleep", lambda _s: None)

    with _patched(fake), pytest.raises(MechRequestRejectedError) as excinfo:
        session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert excinfo.value.error == "requester_busy"
    assert len([c for c in fake.calls if c["method"] == "POST"]) == 4


@pytest.mark.parametrize(
    "first",
    [
        requests.exceptions.ConnectionError("reset"),
        pytest.param("gateway-502", id="502"),
        pytest.param("gateway-504", id="504"),
    ],
)
def test_unknown_outcomes_replay_the_same_signed_body(first: Any) -> None:
    """A dropped connection or a proxy 502/504 may hide a served, charged call."""
    if isinstance(first, str):
        first = _response(
            int(first[-3:]), b"<html>gateway</html>", {"content-type": "text/html"}
        )
    fake = _FakeFacilitator([first, _response(200, _UPSTREAM_BODY)])
    session = _session_with(fake)()

    with _patched(fake):
        response = session.post(f"{_FACILITATOR}/v1beta/models/x", json={"p": 1})

    assert response.status_code == 200
    assert _posted_body(fake, 0)["request_id"] == _posted_body(fake, 1)["request_id"]
