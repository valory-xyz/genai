#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025-2026 Valory AG
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

"""Utility functions for GenAI connection."""

from typing import Any, List, Union, get_args, get_origin

from pydantic import BaseModel


def pydantic_to_gemini_schema(model: Union[type[BaseModel], type[List[Any]]]) -> dict:
    """Convert a Pydantic model (or List[PydanticModel]) into a Gemini-compatible schema."""

    # Handle the simple case — a single Pydantic model
    if isinstance(model, type) and issubclass(model, BaseModel):
        return model.model_json_schema()

    # Handle the case where it's a List[PydanticModel]
    origin = get_origin(model)
    if origin in (list, List):
        args = get_args(model)
        if not args:
            raise TypeError("List[] must specify an inner Pydantic model type.")

        item_model = args[0]
        if not (isinstance(item_model, type) and issubclass(item_model, BaseModel)):
            raise TypeError(f"Expected List[PydanticModel], got List[{item_model}]")

        schema = item_model.model_json_schema()
        return {
            "type": "array",
            "items": {k: v for k, v in schema.items() if k != "$defs"},
            **({"$defs": schema["$defs"]} if "$defs" in schema else {}),
        }

    raise TypeError(f"Expected a Pydantic model or List[PydanticModel], got: {model}")
