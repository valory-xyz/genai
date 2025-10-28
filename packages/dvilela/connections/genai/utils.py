#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2025 Valory AG
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

import inspect
from typing import Any, List, Union, get_args, get_origin

from pydantic import BaseModel


def pydantic_to_gemini_schema(model: Union[type[BaseModel], type[List[Any]]]) -> dict:
    """Convert a Pydantic model (or list thereof) into a Gemini-compatible responseSchema."""

    type_map = {
        str: "STRING",
        int: "INTEGER",
        float: "NUMBER",
        bool: "BOOLEAN",
        list: "ARRAY",
        dict: "OBJECT",
    }

    def resolve_type(py_type: Any) -> dict:
        """Map Python or typing types to Gemini-compatible JSON schema."""
        origin = get_origin(py_type)

        # Handle Optional / Union
        if origin is Union:
            args = [a for a in get_args(py_type) if a is not type(None)]  # noqa: E721
            if args:
                return resolve_type(args[0])
            return {"type": "STRING"}  # fallback

        # Handle nested Pydantic models
        if inspect.isclass(py_type) and issubclass(py_type, BaseModel):
            return pydantic_to_gemini_schema(py_type)

        # Handle lists
        if origin in (list, List):
            item_type = get_args(py_type)[0] if get_args(py_type) else Any
            return {"type": "ARRAY", "items": resolve_type(item_type)}

        # Primitive types
        return {"type": type_map.get(py_type, "STRING")}

    # Handle list[BaseModel] as the top-level schema
    origin = get_origin(model)
    if origin in (list, List):
        item_type = get_args(model)[0]
        return {"type": "ARRAY", "items": pydantic_to_gemini_schema(item_type)}

    # Handle BaseModel class
    if inspect.isclass(model) and issubclass(model, BaseModel):
        properties = {}
        required = []
        for name, field in model.model_fields.items():
            field_schema = resolve_type(field.annotation)
            properties[name] = field_schema
            if field.is_required():
                required.append(name)

        schema = {"type": "OBJECT", "properties": properties}
        if required:
            schema["required"] = required
        return schema

    raise TypeError(f"Unsupported model type: {model!r}")
