"""Configured hand input."""

from config import INPUT_SOURCE


def create_source():
    if INPUT_SOURCE == "wilor":
        from .wilor.source import WilorSource
        return WilorSource()
    if INPUT_SOURCE == "manus":
        from .manus.source import ManusSource
        return ManusSource()
    raise ValueError(f"Unsupported INPUT_SOURCE: {INPUT_SOURCE!r}")
