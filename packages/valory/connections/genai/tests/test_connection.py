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
from typing import Any
from unittest.mock import MagicMock

import google.api_core.exceptions
import pytest

from packages.valory.connections.genai import connection as genai_connection
from packages.valory.connections.genai.connection import (
    GENAI_DIRECT_TIMEOUT_SECONDS,
    GenaiConnection,
)
from packages.valory.connections.x402.clients.base import PaymentError


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

    def _make_x402_stub(self) -> Any:
        """Build a stub with the attributes ``_process_x402_request`` reads."""
        return SimpleNamespace(
            use_x402=True,
            logger=MagicMock(),
            genai_x402_server_base_url="http://x402.example.com",
            connection_private_key=(
                "0x4c0883a69102937d6231471b5dbb6204fe5129617082792ae468d01a3f362318"
            ),
            _eoa_account=MagicMock(),  # x402_requests is monkeypatched away
        )

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
