"""Utilities for initializing the Tonic Textual client."""

from __future__ import annotations

import os
from typing import Any

from langchain_core.utils import convert_to_secret_str
from tonic_textual.redact_api import TextualNer  # type: ignore[import-untyped]


def initialize_client(values: dict[str, Any]) -> dict[str, Any]:
    """Initialize the Tonic Textual client from values or environment."""
    api_key = values.get("tonic_textual_api_key") or os.environ.get(
        "TONIC_TEXTUAL_API_KEY", ""
    )
    values["tonic_textual_api_key"] = convert_to_secret_str(api_key)

    kwargs: dict[str, Any] = {
        "api_key": values["tonic_textual_api_key"].get_secret_value(),
    }
    if values.get("tonic_textual_base_url"):
        kwargs["base_url"] = values["tonic_textual_base_url"]

    values["client"] = TextualNer(**kwargs)
    return values
