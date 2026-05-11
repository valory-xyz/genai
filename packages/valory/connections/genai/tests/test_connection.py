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

"""Tests for the Genai connection.

The default-timeout test for x402 also lives in this file because the
upstream-shape ``packages/valory/connections/x402`` directory is excluded
from this repo's pytest collection (see ``[tool.tomte] pytest_targets_exclude``
in pyproject.toml). The Valory-specific timeout injection is asserted here
where it is collected by the test matrix.
"""

# pylint: disable=protected-access

import asyncio
import base64
import datetime
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import google.api_core.exceptions
import httpx
import pytest
import requests
from eth_account import Account

from packages.valory.connections.genai import connection as genai_connection
from packages.valory.connections.genai.connection import (
    GENAI_DIRECT_TIMEOUT_SECONDS,
    GenaiConnection,
)
from packages.valory.connections.x402.clients.base import (
    PaymentError,
    PaymentResponseDecodeError,
    decode_x_payment_response,
)
from packages.valory.connections.x402.clients.httpx import (
    DEFAULT_X402_HTTPX_TIMEOUT,
    HttpxHooks,
    x402HttpxClient,
)
from packages.valory.connections.x402.clients.requests import (
    DEFAULT_X402_TIMEOUT,
    x402_requests,
)


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


def _fake_super_send(_self_inner: object, _request: object, **kwargs: object) -> object:
    """Capture-friendly stand-in for ``HTTPAdapter.send``."""
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.cookies = {}
    response.is_redirect = False
    response.is_permanent_redirect = False
    response.history = []
    response.url = "http://example.com/"
    response.elapsed = datetime.timedelta(seconds=0)
    response._captured = kwargs  # exposed for the test to read back
    return response


class TestX402DefaultTimeout:
    """Tests covering the default timeout injection on the x402 adapter."""

    def test_default_timeout_is_injected_when_caller_omits(self) -> None:
        """``Session.request`` without a timeout reaches the adapter as the default."""
        session = x402_requests(Account.create())
        captured: dict = {}

        def capturing_send(
            _self_inner: object, request: object, **kwargs: object
        ) -> object:
            captured.update(kwargs)
            return _fake_super_send(_self_inner, request, **kwargs)

        with patch.object(
            requests.adapters.HTTPAdapter,
            "send",
            autospec=True,
            side_effect=capturing_send,
        ):
            session.request("GET", "http://example.com/")
        assert captured["timeout"] == DEFAULT_X402_TIMEOUT

    def test_explicit_caller_timeout_is_preserved(self) -> None:
        """An explicit caller timeout flows through to the adapter unchanged."""
        session = x402_requests(Account.create())
        captured: dict = {}

        def capturing_send(
            _self_inner: object, request: object, **kwargs: object
        ) -> object:
            captured.update(kwargs)
            return _fake_super_send(_self_inner, request, **kwargs)

        with patch.object(
            requests.adapters.HTTPAdapter,
            "send",
            autospec=True,
            side_effect=capturing_send,
        ):
            session.request("GET", "http://example.com/", timeout=5)
        assert captured["timeout"] == 5

    def test_custom_default_timeout_propagates(self) -> None:
        """A caller-provided ``default_timeout`` is mounted on the adapter."""
        session = x402_requests(Account.create(), default_timeout=(2.0, 7.0))
        adapter = session.get_adapter("http://example.com/")
        assert adapter._default_timeout == (2.0, 7.0)


class TestDecodeXPaymentResponse:
    """Tests covering typed errors from ``decode_x_payment_response``."""

    def test_happy_path_returns_decoded_dict(self) -> None:
        """A well-formed header decodes to the expected dict."""
        payload = {"success": True, "transaction": "0xdead", "network": "base"}
        encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
        assert decode_x_payment_response(encoded) == payload

    def test_invalid_base64_raises_payment_response_decode_error(self) -> None:
        """Non-base64 input raises ``PaymentResponseDecodeError``, not bare exception.

        ``binascii.Error`` would otherwise escape and be reported to the
        caller as a generic Genai error, masking that the payment-adapter
        is at fault.
        """
        with pytest.raises(PaymentResponseDecodeError):
            decode_x_payment_response("!!!not-base64!!!")

    def test_invalid_utf8_raises_payment_response_decode_error(self) -> None:
        """Base64 of non-UTF8 bytes raises ``PaymentResponseDecodeError``."""
        encoded = base64.b64encode(b"\xff\xfe\xfd").decode("utf-8")
        with pytest.raises(PaymentResponseDecodeError):
            decode_x_payment_response(encoded)

    def test_non_json_payload_raises_payment_response_decode_error(self) -> None:
        """Valid UTF-8 that is not JSON raises ``PaymentResponseDecodeError``."""
        encoded = base64.b64encode(b"not json at all").decode("utf-8")
        with pytest.raises(PaymentResponseDecodeError):
            decode_x_payment_response(encoded)

    def test_payment_response_decode_error_is_payment_error(self) -> None:
        """The new error type is a ``PaymentError`` so callers' typed catch fires."""
        assert issubclass(PaymentResponseDecodeError, PaymentError)


class TestProcessX402RequestPaymentResponseHeader:
    """Tests covering payment-response header handling in ``_process_x402_request``.

    The earlier code accessed ``payment_response['transaction']`` directly,
    so a successful Gemini call whose payment header was missing the key
    was reported to the user as a Genai error. The fix uses ``.get()`` so
    logging cannot derail a successful response.
    """

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
        """Logging falls back to ``<missing>`` and the call returns the text.

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
        """A ``PaymentError`` is reported under an x402 label, not as a Genai error.

        Drives ``_get_response`` so the typed ``except PaymentError`` branch
        is exercised end-to-end.
        """
        stub = self._make_x402_stub()

        # Bind a stub _process_x402_request onto the namespace so
        # ``_get_response``'s call to ``self._process_x402_request(...)``
        # resolves here. This sidesteps the unrelated pre-existing keyerror
        # path on ``generation_config_kwargs["response_schema"]`` and only
        # exercises the typed-except branch we care about.
        def fake_process(*_a: Any, **_k: Any) -> Any:
            raise PaymentError("Failed to handle payment: boom")

        stub._process_x402_request = fake_process

        body, error = GenaiConnection._get_response(stub, '{"prompt": "hi"}')
        assert error is True
        assert "x402 payment adapter error" in body["error"]
        assert "Genai" not in body["error"]


class TestX402RequestsSecondary402:
    """Tests covering the requests adapter's behaviour on a secondary 402."""

    def test_secondary_402_raises_payment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 402 on the post-payment retry surfaces as ``PaymentError``.

        Before the fix, the retry's 402 body was silently copied to the
        original response and returned, leaving the caller to discover
        the failure by reading the body. Now it raises a typed error
        with a clear message.

        :param monkeypatch: pytest fixture used to stub adapter internals.
        """
        from packages.valory.connections.x402.clients.requests import x402HTTPAdapter
        from packages.valory.connections.x402.types import (
            PaymentRequirements,
            x402PaymentRequiredResponse,
        )

        session = x402_requests(Account.create())
        adapter: x402HTTPAdapter = session.get_adapter("http://example.com/")

        first = MagicMock()
        first.status_code = 402
        accepts = [
            PaymentRequirements(
                scheme="exact",
                network="base",
                max_amount_required="1",
                resource="http://example.com/",
                description="t",
                mime_type="application/json",
                pay_to="0x0000000000000000000000000000000000000000",
                max_timeout_seconds=60,
                asset="0x0000000000000000000000000000000000000000",
            )
        ]
        first_body = x402PaymentRequiredResponse(
            x402_version=1, accepts=accepts, error=""
        )
        first.content = json.dumps(first_body.model_dump(by_alias=True)).encode("utf-8")

        retry = MagicMock()
        retry.status_code = 402
        retry.headers = {}
        retry.content = b'{"x402Version": 1, "accepts": []}'

        call_state = {"n": 0}

        def fake_send(_self: object, _request: object, **_kwargs: object) -> Any:
            call_state["n"] += 1
            return first if call_state["n"] == 1 else retry

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", fake_send)
        monkeypatch.setattr(
            adapter.client,
            "create_payment_header",
            lambda *_a, **_k: "payment-header",
        )

        req = requests.Request("GET", "http://example.com/").prepare()
        with pytest.raises(PaymentError, match="upstream returned 402"):
            adapter.send(req, timeout=5)


class TestX402HttpxRetryTimeout:
    """Tests covering timeout inheritance + secondary-402 handling for httpx."""

    def test_retry_timeout_defaults_to_module_constant(self) -> None:
        """``HttpxHooks`` without an override uses ``DEFAULT_X402_HTTPX_TIMEOUT``."""
        hooks = HttpxHooks(MagicMock())
        assert hooks._retry_timeout.connect == DEFAULT_X402_HTTPX_TIMEOUT[0]
        assert hooks._retry_timeout.read == DEFAULT_X402_HTTPX_TIMEOUT[1]

    def test_retry_timeout_accepts_tuple_override(self) -> None:
        """A caller-supplied (connect, read) tuple flows through to ``Timeout``."""
        hooks = HttpxHooks(MagicMock(), retry_timeout=(3.0, 9.0))
        assert hooks._retry_timeout.connect == 3.0
        assert hooks._retry_timeout.read == 9.0

    def test_retry_timeout_accepts_float_override(self) -> None:
        """A caller-supplied float is treated as connect = read = float."""
        hooks = HttpxHooks(MagicMock(), retry_timeout=4.0)
        assert hooks._retry_timeout.connect == 4.0
        assert hooks._retry_timeout.read == 4.0

    def test_retry_timeout_accepts_httpx_timeout_passthrough(self) -> None:
        """A pre-built ``httpx.Timeout`` is passed through unchanged."""
        explicit = httpx.Timeout(connect=1.0, read=2.0, write=3.0, pool=4.0)
        hooks = HttpxHooks(MagicMock(), retry_timeout=explicit)
        assert hooks._retry_timeout is explicit

    def test_client_class_threads_timeout_to_hooks(self) -> None:
        """``x402HttpxClient`` propagates ``retry_timeout`` into its hook."""
        client = x402HttpxClient(Account.create(), retry_timeout=(2.5, 6.0))
        on_response = client.event_hooks["response"][0]
        # The hook is a bound method on the underlying HttpxHooks instance.
        hooks_instance = on_response.__self__
        assert hooks_instance._retry_timeout.connect == 2.5
        assert hooks_instance._retry_timeout.read == 6.0

    def test_secondary_402_raises_payment_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 402 on the post-payment retry surfaces as ``PaymentError``.

        :param monkeypatch: pytest fixture used to stub the retry client.
        """
        from packages.valory.connections.x402.types import (
            PaymentRequirements,
            x402PaymentRequiredResponse,
        )

        hooks = HttpxHooks(MagicMock())
        first_body = x402PaymentRequiredResponse(
            x402_version=1,
            accepts=[
                PaymentRequirements(
                    scheme="exact",
                    network="base",
                    max_amount_required="1",
                    resource="http://example.com/",
                    description="t",
                    mime_type="application/json",
                    pay_to="0x0000000000000000000000000000000000000000",
                    max_timeout_seconds=60,
                    asset="0x0000000000000000000000000000000000000000",
                )
            ],
            error="",
        )

        first_response = MagicMock(spec=httpx.Response)
        first_response.status_code = 402
        first_response.request = MagicMock(spec=httpx.Request)
        first_response.request.headers = {}
        first_response.aread = MagicMock(
            return_value=asyncio.sleep(0)  # awaitable no-op
        )
        first_response.json.return_value = first_body.model_dump(by_alias=True)
        # Allow attribute assignment for the copy-over branch
        first_response.headers = {}
        first_response._content = b""

        retry_response = MagicMock(spec=httpx.Response)
        retry_response.status_code = 402
        retry_response.headers = {}
        retry_response._content = b""

        class _StubAsyncClient:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs

            async def __aenter__(self) -> "_StubAsyncClient":
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def send(self, _request: object) -> object:
                return retry_response

        monkeypatch.setattr(
            "packages.valory.connections.x402.clients.httpx.AsyncClient",
            _StubAsyncClient,
        )
        hooks.client.select_payment_requirements = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda accepts: accepts[0]
        )
        hooks.client.create_payment_header = MagicMock(  # type: ignore[method-assign]
            return_value="payment-header"
        )

        async def _run() -> None:
            await hooks.on_response(first_response)

        with pytest.raises(PaymentError, match="upstream returned 402"):
            asyncio.run(_run())
