#!/usr/bin/env python3
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

"""Tests for the Genai connection."""

# pylint: disable=protected-access

import base64
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import google.api_core.exceptions
import pytest

from packages.valory.connections.genai import connection as genai_connection
from packages.valory.connections.genai.connection import (
    GENAI_DIRECT_TIMEOUT_SECONDS,
    GenaiConnection,
)
from packages.valory.connections.x402.clients.base import (
    PaymentError,
    PaymentRejectedAfterRetryError,
)
from packages.valory.connections.x402.clients.mech import MechDepositRequiredError


def _make_stub_for_get_response(use_x402: bool = False) -> Any:
    """Build a minimal stub that satisfies ``_get_response``'s instance use."""
    return SimpleNamespace(
        use_x402=use_x402,
        logger=MagicMock(),
    )


class TestGetResponsePayloadValidation:
    """Tests covering payload deserialization inside ``_get_response``."""

    def test_malformed_json_payload_returns_error(self) -> None:
        """A non-JSON SRR payload yields a 'failed to decode' error envelope."""
        stub = _make_stub_for_get_response()
        body, error = GenaiConnection._get_response(stub, "not json")
        assert error is True
        assert "Failed to decode SRR payload as JSON" in body["error"]

    def test_non_dict_decoded_payload_returns_error(self) -> None:
        """A JSON value that decodes to a non-object yields an error envelope."""
        stub = _make_stub_for_get_response()
        body, error = GenaiConnection._get_response(stub, "[1, 2, 3]")
        assert error is True
        assert "must decode to a JSON object" in body["error"]

    def test_missing_required_property_returns_error(self) -> None:
        """A valid JSON object without ``prompt`` is rejected."""
        stub = _make_stub_for_get_response()
        body, error = GenaiConnection._get_response(stub, '{"foo": "bar"}')
        assert error is True
        assert "missing from the request data" in body["error"]


class TestGenerateContentDeadline:
    """Tests covering the SDK-level deadline plumbed through ``_get_response``."""

    def test_timeout_is_forwarded_to_sdk_request_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The deadline reaches the SDK as ``request_options['timeout']``.

        Without this, the deadline is a silent no-op — the SDK falls back
        to its internal default and the connection's worker thread blocks
        for that whole window.

        :param monkeypatch: pytest fixture used to swap ``GenerativeModel``.
        """
        stub = _make_stub_for_get_response()

        captured: dict = {}

        def capture(*_args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(text="ok")

        fake_model = MagicMock()
        fake_model.generate_content = capture
        monkeypatch.setattr(
            genai_connection.genai,
            "GenerativeModel",
            MagicMock(return_value=fake_model),
        )

        body, error = GenaiConnection._get_response(stub, '{"prompt": "x"}')
        assert error is False
        assert body == {"response": "ok"}
        assert captured["request_options"] == {"timeout": GENAI_DIRECT_TIMEOUT_SECONDS}

    def test_deadline_exceeded_becomes_error_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``DeadlineExceeded`` from the SDK is converted into an error envelope.

        The SDK raises this synchronously at the gRPC layer when the
        request_options timeout fires, so this test runs in microseconds
        and exercises the actual production failure path.

        :param monkeypatch: pytest fixture used to swap ``GenerativeModel``.
        """
        stub = _make_stub_for_get_response()

        def raise_deadline(*_args: Any, **_kwargs: Any) -> Any:
            raise google.api_core.exceptions.DeadlineExceeded("504 Deadline Exceeded")

        fake_model = MagicMock()
        fake_model.generate_content = raise_deadline
        monkeypatch.setattr(
            genai_connection.genai,
            "GenerativeModel",
            MagicMock(return_value=fake_model),
        )

        body, error = GenaiConnection._get_response(stub, '{"prompt": "x"}')
        assert error is True
        assert "Deadline Exceeded" in body["error"]


class TestProcessX402RequestPaymentResponseHeader:
    """Tests for ``_process_x402_request``."""

    def _make_x402_stub(self, **overrides: Any) -> Any:
        """Build a stub with the attributes ``_process_x402_request`` reads."""
        stub = SimpleNamespace(
            use_x402=True,
            use_mech_facilitator=False,
            mech_facilitator_base_url="http://facilitator.example",
            mech_chain="optimism",
            mech_safe_addresses={"optimism": "0x" + "11" * 20},
            mech_max_delivery_rate=12000,
            logger=MagicMock(),
            genai_x402_server_base_url="http://x402.example.com",
            connection_private_key=(
                "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
            ),
            _eoa_account=MagicMock(),  # the session factories are monkeypatched away
            _mech_session=None,
        )
        for key, value in overrides.items():
            setattr(stub, key, value)
        # Resolve the session the same way the real connection does.
        stub._paid_session_and_base_url = (
            lambda: GenaiConnection._paid_session_and_base_url(
                cast(GenaiConnection, stub)
            )
        )
        return stub

    def test_payment_header_missing_transaction_does_not_break_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``transaction`` key in the payment header is tolerated.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        stub = self._make_x402_stub()

        encoded_header_without_tx = base64.b64encode(
            json.dumps({"success": True, "network": "base"}).encode("utf-8")
        ).decode("utf-8")

        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}]
        }
        fake_response.headers = {"X-Payment-Response": encoded_header_without_tx}

        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        text, error = GenaiConnection._process_x402_request(
            stub,
            payload={"prompt": "hi"},
            model_name="gemini-2.5-flash",
            generation_config_kwargs={"response_schema": None},
        )
        assert error is False
        assert text == "hello"

    def test_payment_error_is_labeled_as_payment_adapter_error(self) -> None:
        """``PaymentError`` is surfaced under an x402 label, not as a Genai error."""
        stub = self._make_x402_stub()

        def fake_process(*_a: Any, **_k: Any) -> Any:
            raise PaymentError("Failed to handle payment: boom")

        stub._process_x402_request = fake_process

        body, error = GenaiConnection._get_response(stub, '{"prompt": "hi"}')
        assert error is True
        assert "x402 payment adapter error" in body["error"]
        assert "Genai" not in body["error"]

    def test_rejected_payment_reaches_caller_with_corrected_wording(self) -> None:
        """The caller-visible payload says the payment was rejected, not accepted.

        The message is interpolated verbatim into the ``{"error": ...}`` payload
        this connection returns, so the corrected sentence is a contract on the
        consumer side as much as in the log.
        """
        stub = self._make_x402_stub()

        def fake_process(*_a: Any, **_k: Any) -> Any:
            raise PaymentRejectedAfterRetryError(
                status_code=402, body=b'{"error": "insufficient_funds"}'
            )

        stub._process_x402_request = fake_process

        body, error = GenaiConnection._get_response(stub, '{"prompt": "hi"}')
        assert error is True
        assert "the payment was not accepted" in body["error"]
        assert "after payment was accepted" not in body["error"]
        assert "insufficient_funds" not in body["error"]

    def test_plain_prompt_without_schema_reaches_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt-only payload reaches the upstream and forwards temperature.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        stub = self._make_x402_stub()

        gemini_text = "Plain prompt response from Gemini."
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": gemini_text}]}}]
        }
        fake_response.headers = {}  # no payment-response header

        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        # ``self._process_x402_request`` resolves against the stub, not
        # the class. Bind the real implementation with the stub as self.
        stub._process_x402_request = (
            lambda *a, **k: GenaiConnection._process_x402_request(stub, *a, **k)
        )

        body, error = GenaiConnection._get_response(
            stub, '{"prompt": "hi", "temperature": 0.5}'
        )
        assert error is False
        assert body == {"response": gemini_text}

        sent_data = json.loads(fake_session.post.call_args.kwargs["data"])
        assert sent_data["contents"] == [{"parts": [{"text": "hi"}]}]
        assert sent_data["generationConfig"] == {"temperature": 0.5}

    def test_schema_without_mime_type_defaults_to_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A schema with no mime-type defaults to ``application/json``.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        from pydantic import BaseModel

        class _Prediction(BaseModel):
            confidence: float

        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"confidence": 0.9}'}]}}]
        }
        fake_response.headers = {}
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        text, error = GenaiConnection._process_x402_request(
            stub,
            payload={"prompt": "predict"},
            model_name="gemini-2.5-flash",
            generation_config_kwargs={"response_schema": _Prediction},
        )
        assert error is False
        assert text == '{"confidence": 0.9}'

        sent_data = json.loads(fake_session.post.call_args.kwargs["data"])
        assert sent_data["generationConfig"]["response_mime_type"] == "application/json"
        assert "response_json_schema" in sent_data["generationConfig"]

    def test_custom_mime_type_with_schema_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller-provided mime-type survives alongside a schema.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        from pydantic import BaseModel

        class _Prediction(BaseModel):
            confidence: float

        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "confidence=0.9"}]}}]
        }
        fake_response.headers = {}
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        text, error = GenaiConnection._process_x402_request(
            stub,
            payload={"prompt": "predict"},
            model_name="gemini-2.5-flash",
            generation_config_kwargs={
                "response_schema": _Prediction,
                "response_mime_type": "text/plain",
            },
        )
        assert error is False
        assert text == "confidence=0.9"

        sent_data = json.loads(fake_session.post.call_args.kwargs["data"])
        gen_config = sent_data["generationConfig"]
        # Pin the full shape: both keys present, mime-type is the
        # caller's value, schema produced from the Pydantic class.
        assert set(gen_config.keys()) == {
            "response_mime_type",
            "response_json_schema",
        }
        assert gen_config["response_mime_type"] == "text/plain"
        assert gen_config["response_json_schema"] == _Prediction.model_json_schema()

    def test_mime_type_without_schema_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mime-type without a schema reaches the upstream verbatim.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": '{"x": 1}'}]}}]
        }
        fake_response.headers = {}
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        text, error = GenaiConnection._process_x402_request(
            stub,
            payload={"prompt": "free-form JSON please"},
            model_name="gemini-2.5-flash",
            generation_config_kwargs={
                "response_mime_type": "application/json",
                "response_schema": None,
            },
        )
        assert error is False
        assert text == '{"x": 1}'

        sent_data = json.loads(fake_session.post.call_args.kwargs["data"])
        assert sent_data["generationConfig"] == {
            "response_mime_type": "application/json"
        }
        assert "response_json_schema" not in sent_data["generationConfig"]

    def test_upstream_http_error_raises_payment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-2xx response surfaces as ``PaymentError``, not a Genai error.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        import requests as _requests

        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.raise_for_status.side_effect = _requests.exceptions.HTTPError(
            "500 Server Error"
        )
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        with pytest.raises(PaymentError, match="x402 upstream returned HTTP error"):
            GenaiConnection._process_x402_request(
                stub,
                payload={"prompt": "hi"},
                model_name="gemini-2.5-flash",
                generation_config_kwargs={},
            )

    def test_upstream_non_json_body_raises_payment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 200 with a non-JSON body surfaces as ``PaymentError``.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.side_effect = ValueError("Expecting value")
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        with pytest.raises(PaymentError, match="x402 upstream returned non-JSON body"):
            GenaiConnection._process_x402_request(
                stub,
                payload={"prompt": "hi"},
                model_name="gemini-2.5-flash",
                generation_config_kwargs={},
            )

    def test_empty_candidates_does_not_index_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ``candidates`` list raises ``ValueError``, not ``IndexError``.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"candidates": []}
        fake_response.headers = {}
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        with pytest.raises(ValueError, match="Empty response from Genai API"):
            GenaiConnection._process_x402_request(
                stub,
                payload={"prompt": "hi"},
                model_name="gemini-2.5-flash",
                generation_config_kwargs={},
            )

    def test_empty_parts_does_not_index_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ``parts`` list raises ``ValueError``, not ``IndexError``.

        :param monkeypatch: pytest fixture used to stub the x402 session.
        """
        stub = self._make_x402_stub()
        fake_response = MagicMock()
        fake_response.raise_for_status.return_value = None
        fake_response.json.return_value = {"candidates": [{"content": {"parts": []}}]}
        fake_response.headers = {}
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "x402_requests", lambda *_a, **_k: fake_session
        )

        with pytest.raises(ValueError, match="Empty response from Genai API"):
            GenaiConnection._process_x402_request(
                stub,
                payload={"prompt": "hi"},
                model_name="gemini-2.5-flash",
                generation_config_kwargs={},
            )


class TestMechFacilitatorPath:
    """Tests for the mech-marketplace session selection in the genai connection."""

    def _stub(self, **overrides: Any) -> Any:
        return TestProcessX402RequestPaymentResponseHeader()._make_x402_stub(
            **overrides
        )

    def test_flag_off_keeps_the_x402_session_and_proxy_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the flag off nothing changes: x402 session, x402 proxy URL."""
        stub = self._stub(use_mech_facilitator=False)
        seen: dict = {}

        def fake_x402(account: Any, **_k: Any) -> Any:
            seen["x402_account"] = account
            return "x402-session"

        monkeypatch.setattr(genai_connection, "x402_requests", fake_x402)
        monkeypatch.setattr(
            genai_connection,
            "mech_requests",
            lambda *_a, **_k: pytest.fail("mech_requests must not be used"),
        )

        session, base_url = stub._paid_session_and_base_url()

        assert session == "x402-session"
        assert seen["x402_account"] is stub._eoa_account
        assert base_url == "http://x402.example.com"

    def test_flag_on_builds_a_mech_session_against_the_facilitator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the flag on the session is a mech session and paths hang off /v1beta."""
        stub = self._stub(use_mech_facilitator=True, mech_max_delivery_rate=12000)
        seen: dict = {}

        def fake_mech(account: Any, **kwargs: Any) -> Any:
            seen["account"] = account
            seen.update(kwargs)
            return "mech-session"

        monkeypatch.setattr(genai_connection, "mech_requests", fake_mech)
        monkeypatch.setattr(
            genai_connection,
            "x402_requests",
            lambda *_a, **_k: pytest.fail("x402_requests must not be used"),
        )

        session, base_url = stub._paid_session_and_base_url()

        assert session == "mech-session"
        assert seen["account"] is stub._eoa_account
        assert seen["safe_address"] == "0x" + "11" * 20
        assert seen["chain"] == "optimism"
        assert seen["api"] == "chat"
        assert seen["facilitator_base_url"] == "http://facilitator.example"
        assert seen["max_delivery_rate"] == 12000
        assert base_url == "http://facilitator.example/v1beta"

    def test_mech_session_is_built_once_and_reused_across_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two paid requests share one mech session, so its replay memory survives between them."""
        stub = self._stub(use_mech_facilitator=True, mech_max_delivery_rate=12000)
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}]
        }
        fake_response.headers = {}
        sessions: list = []

        def fake_mech(*_a: Any, **_k: Any) -> Any:
            session = MagicMock()
            session.post.return_value = fake_response
            sessions.append(session)
            return session

        monkeypatch.setattr(genai_connection, "mech_requests", fake_mech)

        for _ in range(2):
            text, error = GenaiConnection._process_x402_request(
                stub,
                payload={"prompt": "hi"},
                model_name="gemini-2.5-flash",
                generation_config_kwargs={},
            )
            assert (text, error) == ("hello", False)

        assert len(sessions) == 1
        assert sessions[0].post.call_count == 2
        assert stub._mech_session is sessions[0]

    def test_flag_on_without_a_safe_for_the_chain_is_a_payment_error(self) -> None:
        """Misconfiguration surfaces as a PaymentError, not a KeyError."""
        stub = self._stub(use_mech_facilitator=True, mech_safe_addresses={})

        with pytest.raises(PaymentError):
            stub._paid_session_and_base_url()

    @pytest.mark.parametrize("safe_key", ["optimism", "OPTIMISM"])
    def test_flag_on_matches_the_chain_slug_case_insensitively(
        self, monkeypatch: pytest.MonkeyPatch, safe_key: str
    ) -> None:
        """A mixed-case slug or Safe-map key still resolves and reaches the wire lowercase."""
        stub = self._stub(
            use_mech_facilitator=True,
            mech_chain="Optimism",
            mech_safe_addresses={safe_key: "0x" + "11" * 20},
        )
        seen: dict = {}
        monkeypatch.setattr(
            genai_connection,
            "mech_requests",
            lambda _account, **kwargs: seen.update(kwargs) or "mech-session",
        )

        stub._paid_session_and_base_url()

        assert seen["chain"] == "optimism"
        assert seen["safe_address"] == "0x" + "11" * 20

    @pytest.mark.parametrize(
        ("use_x402", "use_mech", "warns"),
        [(False, True, True), (True, True, False), (False, False, False)],
    )
    def test_mech_flag_without_x402_is_flagged_at_construction(
        self, use_x402: bool, use_mech: bool, warns: bool
    ) -> None:
        """The mech flag only acts on the paid path; setting it alone is a misconfiguration."""
        logger = MagicMock()

        genai_connection._check_payment_flags(logger, use_x402, use_mech, 12000)

        assert logger.warning.called is warns

    def test_mech_flag_without_a_rate_cap_warns_at_construction(self) -> None:
        """Enabling the mech path without a rate cap is called out when the connection starts."""
        logger = MagicMock()

        genai_connection._check_payment_flags(logger, True, True, None)

        assert logger.warning.called is True

    def test_flag_on_without_a_rate_cap_is_a_payment_error(self) -> None:
        """Without a cap the agent would sign any rate the facilitator reports."""
        stub = self._stub(use_mech_facilitator=True, mech_max_delivery_rate=None)

        with pytest.raises(PaymentError, match="mech_max_delivery_rate"):
            stub._paid_session_and_base_url()

    def test_mech_path_does_not_warn_about_a_missing_payment_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mech-settled calls carry no X-Payment-Response; that is not a warning."""
        stub = self._stub(use_mech_facilitator=True)
        fake_response = MagicMock()
        fake_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}}]
        }
        fake_response.headers = {}
        fake_session = MagicMock()
        fake_session.post.return_value = fake_response
        monkeypatch.setattr(
            genai_connection, "mech_requests", lambda *_a, **_k: fake_session
        )

        text, error = GenaiConnection._process_x402_request(
            stub,
            payload={"prompt": "hi"},
            model_name="gemini-2.5-flash",
            generation_config_kwargs={},
        )

        assert (text, error) == ("hello", False)
        stub.logger.warning.assert_not_called()
        posted_url = fake_session.post.call_args.args[0]
        assert posted_url == (
            "http://facilitator.example/v1beta/models/gemini-2.5-flash:generateContent"
        )

    def test_deposit_required_is_returned_as_a_structured_error(self) -> None:
        """The skill needs the shortfall numbers to trigger a top-up."""
        stub = _make_stub_for_get_response(use_x402=True)

        def fake_process(*_a: Any, **_k: Any) -> Any:
            raise MechDepositRequiredError(
                balance=12000, reserved=10000, available=2000, required=10000
            )

        stub._process_x402_request = fake_process

        body, error = GenaiConnection._get_response(stub, '{"prompt": "hi"}')

        assert error is True
        assert body["code"] == "mech_deposit_required"
        assert body["context"] == {
            "balance": 12000,
            "reserved": 10000,
            "available": 2000,
            "required": 10000,
        }
