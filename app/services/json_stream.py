"""Incrementally expose a JSON string field from streamed model output."""

from __future__ import annotations

import re


class JsonStringFieldAccumulator:
    """Return only newly decoded characters for one JSON string field."""

    def __init__(self, field: str):
        self._field = field
        self._source = ""
        self._value = ""

    def feed(self, chunk: str) -> str:
        self._source += chunk
        current = _partial_json_string_value(self._source, self._field)
        if current is None or not current.startswith(self._value):
            return ""
        delta = current[len(self._value) :]
        self._value = current
        return delta


def _partial_json_string_value(source: str, field: str) -> str | None:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', source)
    if match is None:
        return None
    index = match.end()
    result: list[str] = []
    escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < len(source):
        character = source[index]
        if character == '"':
            return "".join(result)
        if character != "\\":
            result.append(character)
            index += 1
            continue
        if index + 1 >= len(source):
            break
        escaped = source[index + 1]
        if escaped == "u":
            digits = source[index + 2 : index + 6]
            if len(digits) < 4:
                break
            try:
                result.append(chr(int(digits, 16)))
            except ValueError:
                break
            index += 6
            continue
        if escaped not in escapes:
            break
        result.append(escapes[escaped])
        index += 2
    return "".join(result)
