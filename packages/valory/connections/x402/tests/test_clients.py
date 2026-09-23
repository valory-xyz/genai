"""Tests for the x402 payment-adapter clients."""

# pylint: disable=protected-access

import asyncio
import base64
import datetime
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import requests
from eth_account import Account

from packages.valory.connections.x402.clients.base import (
    REJECTION_BODY_LOG_LIMIT,
    PaymentError,
    PaymentRejectedAfterRetryError,
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


EXPECTED_REJECTION_MESSAGE = (
    "upstream rejected request with HTTP 402 after the payment header was "
    "attached; the payment was not accepted"
)

# A well-formed x402 payment-required body: the gateway's own reason lives in
# ``error`` and is what a support engineer needs out of a log bundle.
DECODABLE_RETRY_BODY = (
    b'{"x402Version": 1, "accepts": [], "error": "insufficient_funds: '
    b'payer balance below maxAmountRequired"}'
)
GATEWAY_REASON = "insufficient_funds: payer balance below maxAmountRequired"
X402_LOGGER_PREFIX = "packages.valory.connections.x402.clients"


def _payment_required_body() -> bytes:
    """Build the first-402 body both adapters parse before paying.

    :return: the encoded payment-required response.
    """
    from packages.valory.connections.x402.types import (
        PaymentRequirements,
        x402PaymentRequiredResponse,
    )

    body = x402PaymentRequiredResponse(
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
    return json.dumps(body.model_dump(by_alias=True)).encode("utf-8")


def _drive_requests_secondary_402(
    monkeypatch: pytest.MonkeyPatch, retry_body: bytes
) -> None:
    """Drive the requests adapter through a 402-then-402 exchange.

    :param monkeypatch: pytest fixture used to stub adapter internals.
    :param retry_body: body the gateway returns on the paid retry.
    """
    from packages.valory.connections.x402.clients.requests import x402HTTPAdapter

    session = x402_requests(Account.create())
    adapter: x402HTTPAdapter = session.get_adapter("http://example.com/")

    first = MagicMock()
    first.status_code = 402
    first.content = _payment_required_body()

    retry = MagicMock()
    retry.status_code = 402
    retry.headers = {}
    retry.content = retry_body

    send_mock = MagicMock(side_effect=[first, retry])

    def super_send(_self: object, _request: object, **kwargs: object) -> Any:
        return send_mock(_self, _request, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", super_send)
    monkeypatch.setattr(
        adapter.client, "create_payment_header", lambda *_a, **_k: "payment-header"
    )

    req = requests.Request("GET", "http://example.com/").prepare()
    with pytest.raises(PaymentRejectedAfterRetryError):
        adapter.send(req, timeout=5)


def _drive_httpx_secondary_402(
    monkeypatch: pytest.MonkeyPatch, retry_body: bytes
) -> None:
    """Drive the httpx adapter through a 402-then-402 exchange.

    :param monkeypatch: pytest fixture used to stub the retry client.
    :param retry_body: body the gateway returns on the paid retry.
    """
    hooks = HttpxHooks(MagicMock())

    first_response = MagicMock(spec=httpx.Response)
    first_response.status_code = 402
    first_response.request = MagicMock(spec=httpx.Request)
    first_response.request.headers = {}
    first_response.aread = AsyncMock(return_value=None)
    first_response.json.return_value = json.loads(_payment_required_body())
    first_response.headers = {}
    first_response._content = b""

    retry_response = MagicMock(spec=httpx.Response)
    retry_response.status_code = 402
    retry_response.headers = {}
    retry_response.content = retry_body

    class _StubAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            pass

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

    with pytest.raises(PaymentRejectedAfterRetryError):
        asyncio.run(hooks.on_response(first_response))


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
        """Non-base64 input raises ``PaymentResponseDecodeError``."""
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


class TestX402RequestsSecondary402:
    """Tests covering the requests adapter's behaviour on a secondary 402."""

    def test_secondary_402_raises_typed_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second 402 raises ``PaymentRejectedAfterRetryError`` with attached body.

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
        retry.content = b'{"x402Version": 1, "accepts": [], "error": "price changed"}'

        send_mock = MagicMock(side_effect=[first, retry])

        def super_send(_self: object, _request: object, **kwargs: object) -> Any:
            return send_mock(_self, _request, **kwargs)

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", super_send)
        monkeypatch.setattr(
            adapter.client,
            "create_payment_header",
            lambda *_a, **_k: "payment-header",
        )

        req = requests.Request("GET", "http://example.com/").prepare()
        with pytest.raises(PaymentRejectedAfterRetryError) as exc_info:
            adapter.send(req, timeout=5)
        assert exc_info.value.status_code == 402
        assert b"price changed" in exc_info.value.body
        assert "price changed" not in str(exc_info.value)
        assert str(exc_info.value) == EXPECTED_REJECTION_MESSAGE

    def test_gateway_reason_is_logged_at_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The gateway's own reason reaches a WARNING record, not DEBUG only.

        Pearl log bundles are collected at a level that drops DEBUG, which is
        why ZD#1239 carried the symptom and never the cause.

        :param monkeypatch: pytest fixture used to stub adapter internals.
        :param caplog: pytest fixture capturing log records.
        """
        with caplog.at_level(logging.DEBUG):
            _drive_requests_secondary_402(monkeypatch, DECODABLE_RETRY_BODY)

        adapter_records = [
            r for r in caplog.records if r.name.startswith(X402_LOGGER_PREFIX)
        ]
        warnings = [r for r in adapter_records if r.levelno == logging.WARNING]
        assert any(GATEWAY_REASON in r.getMessage() for r in warnings)
        assert not [r for r in adapter_records if r.levelno == logging.DEBUG]

    def test_non_decodable_body_falls_back_to_truncated_repr(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A body that is not an x402 response still produces a WARNING.

        :param monkeypatch: pytest fixture used to stub adapter internals.
        :param caplog: pytest fixture capturing log records.
        """
        with caplog.at_level(logging.WARNING):
            _drive_requests_secondary_402(monkeypatch, b"<html>bad gateway</html>")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("bad gateway" in r.getMessage() for r in warnings)

    def test_oversized_non_decodable_body_is_truncated(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An oversized unparseable body is bounded at the 500-byte limit.

        :param monkeypatch: pytest fixture used to stub adapter internals.
        :param caplog: pytest fixture capturing log records.
        """
        with caplog.at_level(logging.WARNING):
            _drive_requests_secondary_402(
                monkeypatch, b"x" * (REJECTION_BODY_LOG_LIMIT + 100)
            )

        message = [r for r in caplog.records if r.levelno == logging.WARNING][
            -1
        ].getMessage()
        assert "x" * REJECTION_BODY_LOG_LIMIT in message
        assert "x" * (REJECTION_BODY_LOG_LIMIT + 1) not in message

    def test_cancelled_error_propagates_without_wrapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``concurrent.futures.CancelledError`` propagates unwrapped.

        :param monkeypatch: pytest fixture used to stub adapter internals.
        """
        import concurrent.futures

        from packages.valory.connections.x402.clients.requests import x402HTTPAdapter
        from packages.valory.connections.x402.types import (
            PaymentRequirements,
            x402PaymentRequiredResponse,
        )

        session = x402_requests(Account.create())
        adapter: x402HTTPAdapter = session.get_adapter("http://example.com/")

        first = MagicMock()
        first.status_code = 402
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
        first.content = json.dumps(first_body.model_dump(by_alias=True)).encode("utf-8")

        def super_send(_self: object, _request: object, **kwargs: object) -> Any:
            if not super_send.first_done:  # type: ignore[attr-defined]
                super_send.first_done = True  # type: ignore[attr-defined]
                return first
            raise concurrent.futures.CancelledError()

        super_send.first_done = False  # type: ignore[attr-defined]

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", super_send)
        monkeypatch.setattr(
            adapter.client,
            "create_payment_header",
            lambda *_a, **_k: "payment-header",
        )

        req = requests.Request("GET", "http://example.com/").prepare()
        with pytest.raises(concurrent.futures.CancelledError):
            adapter.send(req, timeout=5)
        assert adapter._is_retry is False


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
        hooks_instance = on_response.__self__
        assert hooks_instance._retry_timeout.connect == 2.5
        assert hooks_instance._retry_timeout.read == 6.0

    def test_secondary_402_raises_typed_error_with_body_attribute(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second 402 raises typed error; verifies timeout reaches the retry client.

        :param monkeypatch: pytest fixture used to stub the retry client.
        """
        from packages.valory.connections.x402.types import (
            PaymentRequirements,
            x402PaymentRequiredResponse,
        )

        hooks = HttpxHooks(MagicMock(), retry_timeout=(2.5, 7.5))
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
        first_response.aread = AsyncMock(return_value=None)
        first_response.json.return_value = first_body.model_dump(by_alias=True)
        first_response.headers = {}
        first_response._content = b""

        retry_response = MagicMock(spec=httpx.Response)
        retry_response.status_code = 402
        retry_response.headers = {}
        retry_response.content = b'{"error": "price changed"}'

        captured_kwargs: list = []

        class _StubAsyncClient:
            def __init__(self, **kwargs: Any) -> None:
                captured_kwargs.append(kwargs)

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

        with pytest.raises(PaymentRejectedAfterRetryError) as exc_info:
            asyncio.run(_run())
        assert exc_info.value.status_code == 402
        assert b"price changed" in exc_info.value.body
        assert "price changed" not in str(exc_info.value)
        assert str(exc_info.value) == EXPECTED_REJECTION_MESSAGE

        assert len(captured_kwargs) == 1
        timeout = captured_kwargs[0]["timeout"]
        assert timeout.connect == 2.5
        assert timeout.read == 7.5

    def test_gateway_reason_is_logged_at_warning(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The gateway's own reason reaches a WARNING record, not DEBUG only.

        Pearl log bundles are collected at a level that drops DEBUG, which is
        why ZD#1239 carried the symptom and never the cause.

        :param monkeypatch: pytest fixture used to stub the retry client.
        :param caplog: pytest fixture capturing log records.
        """
        with caplog.at_level(logging.DEBUG):
            _drive_httpx_secondary_402(monkeypatch, DECODABLE_RETRY_BODY)

        adapter_records = [
            r for r in caplog.records if r.name.startswith(X402_LOGGER_PREFIX)
        ]
        warnings = [r for r in adapter_records if r.levelno == logging.WARNING]
        assert any(GATEWAY_REASON in r.getMessage() for r in warnings)
        assert not [r for r in adapter_records if r.levelno == logging.DEBUG]

    def test_non_decodable_body_falls_back_to_truncated_repr(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A body that is not an x402 response still produces a WARNING.

        :param monkeypatch: pytest fixture used to stub the retry client.
        :param caplog: pytest fixture capturing log records.
        """
        with caplog.at_level(logging.WARNING):
            _drive_httpx_secondary_402(monkeypatch, b"<html>bad gateway</html>")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("bad gateway" in r.getMessage() for r in warnings)

    def test_oversized_non_decodable_body_is_truncated(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An oversized unparseable body is bounded at the 500-byte limit.

        :param monkeypatch: pytest fixture used to stub the retry client.
        :param caplog: pytest fixture capturing log records.
        """
        with caplog.at_level(logging.WARNING):
            _drive_httpx_secondary_402(
                monkeypatch, b"x" * (REJECTION_BODY_LOG_LIMIT + 100)
            )

        message = [r for r in caplog.records if r.levelno == logging.WARNING][
            -1
        ].getMessage()
        assert "x" * REJECTION_BODY_LOG_LIMIT in message
        assert "x" * (REJECTION_BODY_LOG_LIMIT + 1) not in message

    def test_consecutive_402s_on_same_client_both_handled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_is_retry`` is cleared after every cycle so the next 402 is handled.

        :param monkeypatch: pytest fixture used to stub the retry client.
        """
        from packages.valory.connections.x402.types import (
            PaymentRequirements,
            x402PaymentRequiredResponse,
        )

        hooks = HttpxHooks(MagicMock())
        body = x402PaymentRequiredResponse(
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

        def _make_first_response() -> Any:
            resp = MagicMock(spec=httpx.Response)
            resp.status_code = 402
            resp.request = MagicMock(spec=httpx.Request)
            resp.request.headers = {}
            resp.aread = AsyncMock(return_value=None)
            resp.json.return_value = body.model_dump(by_alias=True)
            resp.headers = {}
            resp._content = b""
            return resp

        retry_ok = MagicMock(spec=httpx.Response)
        retry_ok.status_code = 200
        retry_ok.headers = {}
        retry_ok._content = b"ok"

        class _StubAsyncClient:
            def __init__(self, **_kwargs: Any) -> None:
                pass

            async def __aenter__(self) -> "_StubAsyncClient":
                return self

            async def __aexit__(self, *_a: Any) -> None:
                return None

            async def send(self, _request: object) -> object:
                return retry_ok

        monkeypatch.setattr(
            "packages.valory.connections.x402.clients.httpx.AsyncClient",
            _StubAsyncClient,
        )
        header_calls: list = []
        hooks.client.select_payment_requirements = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda accepts: accepts[0]
        )
        hooks.client.create_payment_header = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda *_a, **_k: header_calls.append(None) or "header",
        )

        async def _run() -> None:
            await hooks.on_response(_make_first_response())
            await hooks.on_response(_make_first_response())

        asyncio.run(_run())

        assert len(header_calls) == 2
        assert hooks._is_retry is False
