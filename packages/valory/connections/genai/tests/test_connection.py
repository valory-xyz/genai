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

import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import google.api_core.exceptions
import pytest
import requests
from eth_account import Account

from packages.valory.connections.genai import connection as genai_connection
from packages.valory.connections.genai.connection import (
    GENAI_DIRECT_TIMEOUT_SECONDS,
    GenaiConnection,
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
