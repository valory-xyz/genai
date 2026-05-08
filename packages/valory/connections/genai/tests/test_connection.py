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

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from packages.valory.connections.genai import connection as genai_connection
from packages.valory.connections.genai.connection import GenaiConnection


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
    """Tests covering the synchronous SDK call's client-side deadline."""

    def test_generate_content_returns_normally_within_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A fast SDK reply propagates through ``_get_response`` unchanged."""
        stub = _make_stub_for_get_response()

        fake_response = SimpleNamespace(text="ok")
        fake_model = MagicMock()
        fake_model.generate_content = MagicMock(return_value=fake_response)
        monkeypatch.setattr(
            genai_connection.genai,
            "GenerativeModel",
            MagicMock(return_value=fake_model),
        )

        body, error = GenaiConnection._get_response(stub, '{"prompt": "x"}')
        assert error is False
        assert body == {"response": "ok"}

    def test_generate_content_respects_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SDK call slower than the deadline is converted to an error envelope."""
        stub = _make_stub_for_get_response()

        # Drop the deadline so the slow-path test runs quickly.
        monkeypatch.setattr(genai_connection, "GENAI_DIRECT_TIMEOUT_SECONDS", 0.05)

        def slow_generate(*_args: Any, **_kwargs: Any) -> Any:
            time.sleep(1.0)
            return SimpleNamespace(text="late")

        fake_model = MagicMock()
        fake_model.generate_content = slow_generate
        monkeypatch.setattr(
            genai_connection.genai,
            "GenerativeModel",
            MagicMock(return_value=fake_model),
        )

        body, error = GenaiConnection._get_response(stub, '{"prompt": "x"}')
        assert error is True
        # The outer except wraps the TimeoutError in "Exception while
        # calling Genai: ..."; the inner message identifies the deadline.
        assert "deadline" in body["error"]
