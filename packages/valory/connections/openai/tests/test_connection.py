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

"""Tests for the OpenAI connection."""

# pylint: disable=protected-access

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from packages.valory.connections.openai import connection as openai_connection
from packages.valory.connections.openai.connection import OpenaiConnection


def _make_stub_for_staging() -> Any:
    """Build a minimal stub for ``_get_response`` exercising the staging API."""
    return SimpleNamespace(
        openai_settings={
            "engine": "gpt-3.5-turbo",
            "max_tokens": 100,
            "temperature": 0.0,
            "request_timeout": 5.0,
            "use_openai_staging_api": True,
            "openai_staging_api": "http://staging.example.com/",
            "openai_api_key": "unused",
        },
        logger=MagicMock(),
    )


class TestStagingApiFailureModes:
    """Tests covering uncaught error paths in the OpenAI staging API branch.

    Before this fix, ``requests.exceptions.RequestException``,
    ``ValueError`` (which JSONDecodeError subclasses), and ``KeyError``
    from the staging branch escaped ``_get_response`` and ``on_send``.
    The pool task callback logged but never sent a response envelope,
    so the calling skill hung until its own round timeout.
    """

    def test_network_failure_returns_error_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``RequestException`` is converted into a user-facing error string.

        :param monkeypatch: pytest fixture used to stub ``requests.post``.
        """
        stub = _make_stub_for_staging()

        def raise_request_exc(*_args: Any, **_kwargs: Any) -> Any:
            raise requests.exceptions.ConnectTimeout("connection timed out")

        monkeypatch.setattr(openai_connection.requests, "post", raise_request_exc)

        result = OpenaiConnection._get_response(stub, "irrelevant", {})
        assert result == "OpenAI staging API request error"

    def test_non_json_body_returns_error_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-JSON 200 body is converted into a user-facing error string.

        :param monkeypatch: pytest fixture used to stub ``requests.post``.
        """
        stub = _make_stub_for_staging()
        response = MagicMock()
        response.json.side_effect = requests.exceptions.JSONDecodeError(
            "Expecting value", "doc", 0
        )
        monkeypatch.setattr(
            openai_connection.requests, "post", lambda *a, **k: response
        )

        result = OpenaiConnection._get_response(stub, "irrelevant", {})
        assert result == "OpenAI staging API decode error"

    def test_missing_text_key_returns_error_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JSON 200 body without ``text`` is converted into an error string.

        :param monkeypatch: pytest fixture used to stub ``requests.post``.
        """
        stub = _make_stub_for_staging()
        response = MagicMock()
        response.json.return_value = {"choices": [{"content": "missing the right key"}]}
        monkeypatch.setattr(
            openai_connection.requests, "post", lambda *a, **k: response
        )

        result = OpenaiConnection._get_response(stub, "irrelevant", {})
        assert result == "OpenAI staging API schema error"

    def test_happy_path_still_returns_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A well-formed JSON 200 body still returns the ``text`` value.

        :param monkeypatch: pytest fixture used to stub ``requests.post``.
        """
        stub = _make_stub_for_staging()
        response = MagicMock()
        response.json.return_value = {"text": "hello"}
        monkeypatch.setattr(
            openai_connection.requests, "post", lambda *a, **k: response
        )

        result = OpenaiConnection._get_response(stub, "irrelevant", {})
        assert result == "hello"
