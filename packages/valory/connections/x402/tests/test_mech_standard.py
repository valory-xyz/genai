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

"""Tests for the standard off-chain mech request adapter.

Only ``HTTPAdapter.send`` is replaced, by a fake facilitator that checks
every posted form the way the facilitator does (CID of ``ipfs_data``,
request id, Safe signature), so the adapter's own building, signing,
fallback and polling all run for real.
"""

# pylint: disable=protected-access,redefined-outer-name

import base64
import json
import time
from email.utils import formatdate
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests
from eth_account import Account
from eth_utils import to_checksum_address

from packages.valory.connections.x402.clients import mech_standard as std
from packages.valory.connections.x402.clients.mech import (
    EXPLICIT_NONCE_DISABLED_REASON,
    MechDepositRequiredError,
    MechOutcomeUnknownError,
    MechRequestRejectedError,
)
from packages.valory.connections.x402.clients.mech_standard import (
    FACILITATOR_MECHS,
    PAYMENT_TYPE_NATIVE,
    PAYMENT_TYPE_USDC,
    REASON_BELOW_MINIMUM,
    REASON_DEADLINE_TOO_SOON,
    REASON_NONCE_BELOW,
    MechLimitBelowMinimumError,
    MechRequestFailedError,
    default_limit,
    mech_standard_requests,
    minimum_limit,
)
from packages.valory.connections.x402.mech_signing import (
    MAX_CID_BLOCK_BYTES,
    cidv1_digest,
    compute_domain_separator,
    compute_explicit_nonce_domain_separator,
    compute_safe_message_hash,
    derive_explicit_request_id,
    derive_request_id,
)

_BASE = "https://facilitator.example"
_CHAIN = "optimism"
_SAFE = to_checksum_address("0x0000000000000000000000000000000000000111")
_ACCOUNT = Account.from_key("0x" + "42" * 32)
_GEMINI_URL = f"{_BASE}/v1beta/models/gemini-2.5-flash:generateContent"
_GEMINI_BODY = {"contents": [{"parts": [{"text": "hello"}]}]}
_GEMINI_ANSWER = '{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]}'
_COINGECKO_URL = f"{_BASE}/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
_COINGECKO_ANSWER = '{"bitcoin":{"usd":1}}'
_CHARGED = 1234


# ---------------------------------------------------------------------------
# CID and request id: pinned to mech-client's and go-ipfs's vectors.
# ---------------------------------------------------------------------------


def _cid_string(content: bytes) -> str:
    raw = bytes([0x01, 0x70, 0x12, 0x20]) + cidv1_digest(content)
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


@pytest.mark.parametrize(
    "content,expected",
    [
        (b"", "bafybeif7ztnhq65lumvvtr4ekcwd2ifwgm3awq4zfr3srh462rwyinlb4y"),
        (b"hello", "bafybeid3weurg3gvyoi7nisadzolomlvoxoppe2sesktnpvdve3256n5tq"),
        (
            b'{"requestId":"42","result":"ok"}',
            "bafybeihjg4w34tu2zmlriemthl24wdynhfx2rpsiflv7wxmb4lsk34etlu",
        ),
        (
            b"x" * 100000,
            "bafybeiay23kics7rguz4kaxxmyz7d6bciozygbi27wllerri6dpqenuifi",
        ),
    ],
)
def test_cid_matches_ipfs_add_output(content: bytes, expected: str) -> None:
    """The local CID is what ``ipfs add --cid-version=1`` gives, as on mech-client."""
    assert _cid_string(content) == expected


def test_cid_refuses_content_over_one_block() -> None:
    """Content past one block would need a DAG, so it is refused, not mis-hashed."""
    with pytest.raises(ValueError, match="single-block"):
        cidv1_digest(b"a" * (MAX_CID_BLOCK_BYTES + 1))


def test_sequential_request_id_matches_mech_client_vector() -> None:
    """Same inputs as mech-client's golden vector from the Gnosis marketplace."""
    marketplace = "0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB"
    assert (
        compute_domain_separator(100, marketplace).hex()
        == "58fbb2508b962bcf6e2708fdfc23222115504128df851ae75ef8c66f2e0bdade"
    )
    request_id = derive_request_id(
        marketplace_address=marketplace,
        mech_address="0x" + "11" * 20,
        requester="0x" + "22" * 20,
        data=bytes.fromhex("33" * 32),
        delivery_rate=7,
        payment_type=bytes.fromhex("44" * 32),
        nonce=9,
        chain_id=100,
    )
    assert (
        request_id.hex()
        == "512e9b1dd2d2d253ba226f6d47af046720b1d97a09ae71e4ecc3f640f33bf36b"
    )


@pytest.mark.parametrize(
    "chain_id,marketplace,expected",
    [
        (
            100,
            "0x735FAAb1c4Ec41128c367AFb5c3baC73509f70bB",
            "0a88545ee8084de72517271130ed61d031c1b1cf48766d523f74a7df2b564b96",
        ),
        (
            137,
            "0x343F2B005cF6D70bA610CD9F1F1927049414B582",
            "cc69de6a75318009b6513d0a69f8f7fe753d9979e76ee68a0c8910be4c14f4e5",
        ),
    ],
)
def test_explicit_request_id_matches_mech_client_vectors(
    chain_id: int, marketplace: str, expected: str
) -> None:
    """Same inputs as mech-client's explicit-nonce vectors from deployed marketplaces."""
    request_id = derive_explicit_request_id(
        domain_separator=compute_explicit_nonce_domain_separator(chain_id, marketplace),
        mech_address="0x" + "11" * 20,
        requester="0x" + "22" * 20,
        data=bytes.fromhex("33" * 32),
        max_delivery_rate=7,
        payment_type=bytes.fromhex("44" * 32),
        nonce=9,
        deadline=1_900_000_000,
    )
    assert request_id.hex() == expected


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,payment_type,default,minimum",
    [
        ("chat", PAYMENT_TYPE_USDC, 30_000, 15_000),
        ("chat", PAYMENT_TYPE_NATIVE, 3 * 10**16, 15 * 10**15),
        ("coingecko", PAYMENT_TYPE_USDC, 2_000, 1_000),
        ("coingecko", PAYMENT_TYPE_NATIVE, 2 * 10**15, 10**15),
    ],
)
def test_default_limit_is_twice_the_minimum_in_the_mech_asset(
    tool: str, payment_type: bytes, default: int, minimum: int
) -> None:
    """Defaults are 3 cents for chat and 0.002 for CoinGecko, in the asset's units."""
    assert default_limit(tool, payment_type) == default
    assert minimum_limit(tool, payment_type) == minimum


def test_unknown_payment_type_is_not_priced() -> None:
    """A limit is never guessed for an asset whose decimals are unknown."""
    with pytest.raises(ValueError, match="not priced"):
        default_limit("chat", b"\x01" * 32)


def test_limit_below_minimum_is_refused_before_signing() -> None:
    """A configured limit the facilitator would refuse never reaches it."""
    with pytest.raises(ValueError, match="below the facilitator's minimum"):
        mech_standard_requests(
            _ACCOUNT,
            safe_address=_SAFE,
            chain=_CHAIN,
            api="chat",
            facilitator_base_url=_BASE,
            delivery_rate_limit=14_999,
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"chain": "arbitrum"}, "no facilitator mech"),
        ({"api": "openai"}, "unknown mech tool"),
        ({"nonce_mode": "random"}, "unknown nonce_mode"),
        ({"mech_address": "0x" + "99" * 20}, "payment type must be configured"),
        ({"payment_type": b"\x01" * 31}, "32 bytes"),
    ],
)
def test_construction_refuses_unusable_settings(
    kwargs: Dict[str, Any], match: str
) -> None:
    """Settings that could only produce refused or mis-signed requests fail early."""
    settings: Dict[str, Any] = {
        "safe_address": _SAFE,
        "chain": _CHAIN,
        "api": "chat",
        "facilitator_base_url": _BASE,
        **kwargs,
    }
    with pytest.raises(ValueError, match=match):
        mech_standard_requests(_ACCOUNT, **settings)


# ---------------------------------------------------------------------------
# A fake facilitator that checks posted forms like the real one.
# ---------------------------------------------------------------------------


def _resp(
    status: int, payload: Any, headers: Optional[Dict[str, str]] = None
) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode("utf-8")
    response.headers.update({"content-type": "application/json", **(headers or {})})
    return response


def _accepted(form: Dict[str, str]) -> requests.Response:
    return _resp(200, {"request_id": form["request_id"]})


def _delivered(request_id: str, result: str = _GEMINI_ANSWER) -> requests.Response:
    return _resp(
        200,
        {
            "request_id": request_id,
            "status": "ok",
            "content_cid": "bafy",
            "response": {"requestId": int(request_id), "result": result},
            "delivery_rate": _CHARGED,
        },
    )


class _Facilitator:
    """Scripted answers, after checking each form as the facilitator does."""

    def __init__(self, chain: str = _CHAIN) -> None:
        self.mech = FACILITATOR_MECHS[chain]
        self.forms: List[Dict[str, str]] = []
        self.polls: List[Dict[str, str]] = []
        self.post_script: List[Any] = []
        self.poll_script: List[Any] = []

    def _check(self, form: Dict[str, str]) -> None:
        data = bytes.fromhex(form["ipfs_hash"][2:])
        assert data == cidv1_digest(form["ipfs_data"].encode("utf-8"))
        assert form["sender"] == _SAFE
        common = {
            "mech_address": self.mech.mech_address,
            "requester": _SAFE,
            "data": data,
            "payment_type": self.mech.payment_type,
            "nonce": int(form["nonce"]),
        }
        if form.get("nonce_mode") == "explicit":
            request_id = derive_explicit_request_id(
                domain_separator=compute_explicit_nonce_domain_separator(
                    self.mech.chain_id, self.mech.marketplace_address
                ),
                max_delivery_rate=int(form["delivery_rate"]),
                deadline=int(form["deadline"]),
                **common,
            )
        else:
            assert "deadline" not in form
            request_id = derive_request_id(
                marketplace_address=self.mech.marketplace_address,
                delivery_rate=int(form["delivery_rate"]),
                chain_id=self.mech.chain_id,
                **common,
            )
        assert int(form["request_id"]) == int.from_bytes(request_id, "big")
        digest = compute_safe_message_hash(request_id, _SAFE, self.mech.chain_id)
        signature = bytes.fromhex(form["signature"][2:])
        signer = Account._recover_hash(digest, signature=signature)
        assert signer == _ACCOUNT.address

    def send(self, _adapter: Any, request: requests.PreparedRequest, **_kw: Any):
        parts = urlsplit(str(request.url))
        if request.method == "POST":
            assert parts.path == f"/mech/offchain/{_CHAIN}/send_signed_requests"
            body = request.body
            text = body.decode() if isinstance(body, bytes) else str(body)
            form = dict(parse_qsl(text, keep_blank_values=True))
            self._check(form)
            self.forms.append(form)
            item = self.post_script.pop(0) if self.post_script else _accepted
            if isinstance(item, Exception):
                raise item
            return item(form) if callable(item) else item
        assert parts.path == f"/mech/offchain/{_CHAIN}/fetch_offchain_info"
        params = dict(parse_qsl(parts.query))
        self.polls.append(params)
        item = self.poll_script.pop(0) if self.poll_script else None
        if item is None:
            return _delivered(params["request_id"])
        if isinstance(item, Exception):
            raise item
        return item(params["request_id"]) if callable(item) else item


@pytest.fixture
def facilitator() -> Iterator[_Facilitator]:
    """Route every HTTP call of the adapter to a fresh fake facilitator."""
    fake = _Facilitator()
    std.standard_explicit_nonce_refusals().clear()
    std.reset_sequential_nonces()
    with patch.object(
        requests.adapters.HTTPAdapter,
        "send",
        lambda adapter, request, **kw: fake.send(adapter, request, **kw),
    ):
        yield fake
    std.standard_explicit_nonce_refusals().clear()
    std.reset_sequential_nonces()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> List[float]:
    """Record waits instead of sleeping."""
    waited: List[float] = []
    monkeypatch.setattr(std.time, "sleep", waited.append)
    return waited


def _session(api: str = "chat", **kwargs: Any) -> requests.Session:
    return mech_standard_requests(
        _ACCOUNT,
        safe_address=_SAFE,
        chain=_CHAIN,
        api=api,
        facilitator_base_url=_BASE,
        **kwargs,
    )


def _chat(session: requests.Session) -> requests.Response:
    return session.post(_GEMINI_URL, data=json.dumps(_GEMINI_BODY))


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def test_chat_call_becomes_a_signed_explicit_chat_request(
    facilitator: _Facilitator,
) -> None:
    """The Gemini body and model are the prompt; the default limit is signed."""
    before = int(time.time())
    response = _chat(_session())
    (form,) = facilitator.forms
    metadata = json.loads(form["ipfs_data"])
    assert metadata["tool"] == "chat"
    assert json.loads(metadata["prompt"]) == {
        **_GEMINI_BODY,
        "model": "gemini-2.5-flash",
    }
    assert form["nonce_mode"] == "explicit"
    assert form["delivery_rate"] == "30000"
    assert before + 900 <= int(form["deadline"]) <= int(time.time()) + 900
    assert response.status_code == 200
    assert response.text == _GEMINI_ANSWER
    assert response.headers["X-Mech-Request-Id"] == form["request_id"]
    assert response.headers["X-Mech-Delivery-Rate"] == str(_CHARGED)


def test_coingecko_get_becomes_a_path_and_query_prompt(
    facilitator: _Facilitator,
) -> None:
    """The path and query of the GET are the prompt the CoinGecko tool reads."""
    facilitator.poll_script = [lambda rid: _delivered(rid, _COINGECKO_ANSWER)]
    response = _session(api="coingecko").get(_COINGECKO_URL)
    (form,) = facilitator.forms
    metadata = json.loads(form["ipfs_data"])
    assert metadata["tool"] == "coingecko"
    assert json.loads(metadata["prompt"]) == {
        "path": "api/v3/simple/price",
        "query": {"ids": "bitcoin", "vs_currencies": "usd"},
    }
    assert form["delivery_rate"] == "2000"
    assert response.json() == {"bitcoin": {"usd": 1}}


def test_each_request_signs_a_new_nonce_and_metadata(facilitator: _Facilitator) -> None:
    """Two identical calls are two requests with different ids."""
    session = _session()
    _chat(session)
    _chat(session)
    first, second = facilitator.forms
    assert first["nonce"] != second["nonce"]
    assert first["request_id"] != second["request_id"]


def test_configured_limit_is_the_signed_rate(facilitator: _Facilitator) -> None:
    """A configured limit replaces the default in the signed request."""
    _chat(_session(delivery_rate_limit=20_000))
    assert facilitator.forms[0]["delivery_rate"] == "20000"


@pytest.mark.parametrize(
    "method,url",
    [
        ("GET", _GEMINI_URL),
        ("POST", f"{_BASE}/v1beta/models/gemini-2.5-flash:countTokens"),
        ("POST", "https://elsewhere.example/v1beta/models/m:generateContent"),
    ],
)
def test_calls_the_tool_does_not_serve_are_refused_unsigned(
    facilitator: _Facilitator, method: str, url: str
) -> None:
    """Nothing is signed for a call that is not one chat request to the facilitator."""
    with pytest.raises(Exception, match="generateContent|not the facilitator"):
        _session().request(method, url, data=json.dumps(_GEMINI_BODY))
    assert not facilitator.forms


# ---------------------------------------------------------------------------
# Polling and results
# ---------------------------------------------------------------------------


def test_held_poll_is_asked_again_until_the_answer_arrives(
    facilitator: _Facilitator, sleeps: List[float]
) -> None:
    """An empty answer means still running; the poll is repeated for the same id."""
    facilitator.poll_script = [_resp(200, {}), _resp(200, {}), None]
    response = _chat(_session())
    assert response.text == _GEMINI_ANSWER
    request_id = facilitator.forms[0]["request_id"]
    assert [p["request_id"] for p in facilitator.polls] == [request_id] * 3
    # Answered at once, so the adapter paced the polls itself.
    assert len(sleeps) == 2 and all(0 < s <= 2.0 for s in sleeps)


def test_poll_survives_a_dropped_connection(
    facilitator: _Facilitator, sleeps: List[float]
) -> None:
    """A lost poll is asked again; the request is not re-signed."""
    facilitator.poll_script = [requests.exceptions.ConnectionError("reset"), None]
    assert _chat(_session()).text == _GEMINI_ANSWER
    assert len(facilitator.forms) == 1
    assert sleeps == [2.0]


@pytest.mark.parametrize("fault", ["client", "mech"])
def test_failed_result_names_who_caused_it(
    facilitator: _Facilitator, fault: str
) -> None:
    """A delivered failure is raised with its fault and charge, not returned as data."""
    facilitator.poll_script = [
        lambda rid: _resp(
            200,
            {
                "request_id": rid,
                "status": "failed",
                "fault": fault,
                "error": "upstream answered HTTP 400: bad",
                "response": {"result": None},
                "delivery_rate": 30000 if fault == "client" else 1,
            },
        )
    ]
    with pytest.raises(MechRequestFailedError) as info:
        _chat(_session())
    assert info.value.fault == fault
    assert info.value.delivery_rate == (30000 if fault == "client" else 1)
    assert str(info.value.request_id) == facilitator.forms[0]["request_id"]


def test_undelivered_request_is_a_rejection(facilitator: _Facilitator) -> None:
    """A run cut off on the facilitator's side is reported as not delivered."""
    facilitator.poll_script = [
        _resp(200, {"status": "rejected", "reason": "request was not completed"})
    ]
    with pytest.raises(MechRequestRejectedError, match="not completed"):
        _chat(_session())


def test_delivery_without_a_result_is_refused(facilitator: _Facilitator) -> None:
    """An ok answer that carries no result text is not passed on as success."""
    facilitator.poll_script = [
        lambda rid: _resp(200, {"request_id": rid, "status": "ok", "response": {}})
    ]
    with pytest.raises(MechRequestRejectedError, match="without a result"):
        _chat(_session())


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_limit_below_minimum_answer_is_typed(facilitator: _Facilitator) -> None:
    """The facilitator's 400 for a low limit carries its minimum to the caller."""
    facilitator.post_script = [
        _resp(
            400,
            {
                "status": "rejected",
                "reason": REASON_BELOW_MINIMUM,
                "minimum_cap": "16000",
            },
        )
    ]
    with pytest.raises(MechLimitBelowMinimumError) as info:
        _chat(_session())
    assert (info.value.limit, info.value.minimum) == (30000, 16000)
    assert not facilitator.polls


def test_payment_challenge_becomes_a_deposit_error(facilitator: _Facilitator) -> None:
    """``required`` covers this request and the unsettled ones; the error splits them."""
    facilitator.post_script = [
        _resp(
            402,
            {
                "status": "rejected",
                "reason": "insufficient balance",
                "currentBalance": "40000",
                "required": "70000",
            },
        )
    ]
    with pytest.raises(MechDepositRequiredError) as info:
        _chat(_session())
    error = info.value
    assert (error.balance, error.reserved, error.available, error.required) == (
        40000,
        40000,
        0,
        30000,
    )


def test_other_refusal_is_raised_with_its_reason(facilitator: _Facilitator) -> None:
    """A refusal that is not retryable is surfaced as is."""
    facilitator.post_script = [
        _resp(
            400, {"status": "rejected", "reason": "chat prompt fields are not served"}
        )
    ]
    with pytest.raises(MechRequestRejectedError, match="not served"):
        _chat(_session())


def test_rate_limit_waits_and_resends_the_same_form(
    facilitator: _Facilitator, sleeps: List[float]
) -> None:
    """A 429 is waited out for its Retry-After and the same signed form sent again."""
    facilitator.post_script = [
        _resp(
            429, {"status": "rejected", "reason": "rate limited"}, {"Retry-After": "3"}
        ),
        _accepted,
    ]
    _chat(_session())
    first, second = facilitator.forms
    assert first == second
    assert sleeps[0] == 3.0


def test_lost_post_resends_the_same_form(facilitator: _Facilitator) -> None:
    """When an answer is lost the identical form goes again, so it is not paid twice."""
    facilitator.post_script = [
        requests.exceptions.ConnectionError("reset"),
        _resp(502, {}),
        _accepted,
    ]
    _chat(_session())
    assert len(facilitator.forms) == 3
    assert facilitator.forms[0] == facilitator.forms[1] == facilitator.forms[2]


def test_unanswered_request_is_asked_about_on_the_next_identical_call(
    facilitator: _Facilitator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request whose answer never came is re-sent, not re-signed, next time."""
    session = _session(total_deadline_secs=1.0)
    clock = {"now": 1000.0}
    monkeypatch.setattr(std.time, "monotonic", lambda: clock["now"])

    def _late(_rid: str) -> requests.Response:
        clock["now"] += 5.0
        return _resp(200, {})

    facilitator.poll_script = [_late]
    with pytest.raises(MechOutcomeUnknownError):
        _chat(session)
    session.adapters["https://"].total_deadline_secs = 300.0
    response = _chat(session)
    first, second = facilitator.forms
    assert first == second
    assert response.headers["X-Mech-Request-Id"] == first["request_id"]


# ---------------------------------------------------------------------------
# Nonce modes
# ---------------------------------------------------------------------------


def _explicit_disabled(_form: Dict[str, str]) -> requests.Response:
    return _resp(503, {"status": "rejected", "reason": EXPLICIT_NONCE_DISABLED_REASON})


def _nonce_below(expected: int):
    def _answer(_form: Dict[str, str]) -> requests.Response:
        return _resp(
            401,
            {
                "status": "rejected",
                "reason": REASON_NONCE_BELOW,
                "expected_nonce": str(expected),
            },
        )

    return _answer


def test_explicit_refusal_falls_back_to_the_sequential_nonce(
    facilitator: _Facilitator,
) -> None:
    """On the facilitator's 503 the call is re-signed sequentially at its expected nonce."""
    facilitator.post_script = [_explicit_disabled, _nonce_below(7), _accepted]
    response = _chat(_session())
    explicit, guess, sequential = facilitator.forms
    assert explicit["nonce_mode"] == "explicit"
    assert "nonce_mode" not in guess and guess["nonce"] == "0"
    assert "nonce_mode" not in sequential and sequential["nonce"] == "7"
    assert response.headers["X-Mech-Request-Id"] == sequential["request_id"]


def test_refusal_is_remembered_so_the_next_call_goes_sequential(
    facilitator: _Facilitator,
) -> None:
    """After one refusal the next call does not try explicit again straight away."""
    session = _session()
    facilitator.post_script = [_explicit_disabled, _nonce_below(7), _accepted]
    _chat(session)
    _chat(session)
    last = facilitator.forms[-1]
    assert "nonce_mode" not in last
    assert last["nonce"] == "8"
    assert len(facilitator.forms) == 4


def test_other_503_is_not_taken_for_an_explicit_refusal(
    facilitator: _Facilitator, sleeps: List[float]
) -> None:
    """A busy facilitator is waited on; the call stays explicit."""
    facilitator.post_script = [
        _resp(
            503,
            {"status": "rejected", "reason": "eip1271 isValidSignature call timed out"},
        ),
        _accepted,
    ]
    _chat(_session())
    assert all(form["nonce_mode"] == "explicit" for form in facilitator.forms)
    assert len(sleeps) == 1
    assert not std.standard_explicit_nonce_refusals().refused(_BASE, _CHAIN)


def test_sequential_mode_gives_up_after_bounded_nonce_corrections(
    facilitator: _Facilitator,
) -> None:
    """A facilitator that keeps moving the nonce is not chased forever."""
    facilitator.post_script = [_nonce_below(n) for n in range(1, 10)]
    with pytest.raises(MechRequestRejectedError, match="nonce below"):
        _chat(_session(nonce_mode="sequential"))
    assert len(facilitator.forms) == std.DEFAULT_NONCE_RETRIES + 1


def test_deadline_too_close_resigns_from_the_facilitators_clock(
    facilitator: _Facilitator,
) -> None:
    """A facilitator clock ahead of ours is used for the second signature."""
    ahead = int(time.time()) + 3600

    def _too_soon(_form: Dict[str, str]) -> requests.Response:
        return _resp(
            401,
            {
                "status": "rejected",
                "reason": REASON_DEADLINE_TOO_SOON,
                "context": {"now": str(ahead)},
            },
        )

    facilitator.post_script = [_too_soon, _accepted]
    _chat(_session())
    first, second = facilitator.forms
    assert int(second["deadline"]) >= ahead + 900 > int(first["deadline"])


def test_date_header_sets_the_clock_explicit_deadlines_use(
    facilitator: _Facilitator,
) -> None:
    """Deadlines are signed from the facilitator's clock once one Date is seen."""
    ahead = time.time() + 7200
    facilitator.post_script = [
        lambda form: _resp(
            200,
            {"request_id": form["request_id"]},
            {"Date": formatdate(ahead, usegmt=True)},
        ),
        _accepted,
    ]
    session = _session()
    _chat(session)
    _chat(session)
    assert int(facilitator.forms[1]["deadline"]) >= int(ahead) + 900 - 5


def test_gnosis_signs_in_the_native_coin() -> None:
    """On Gnosis the facilitator mech is paid in xDAI, so the limit has 18 decimals."""
    session = mech_standard_requests(
        _ACCOUNT,
        safe_address=_SAFE,
        chain="gnosis",
        api="chat",
        facilitator_base_url=_BASE,
    )
    adapter = session.adapters["https://"]
    assert adapter.delivery_rate_limit == 3 * 10**16
    assert adapter.payment_type == PAYMENT_TYPE_NATIVE
