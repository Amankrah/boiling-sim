"""Strip JSONC-style `//` and `#` line comments; `//` inside `"..."` is preserved."""

from __future__ import annotations

import json


def strip_hash_comments(text: str) -> str:
    out_lines: list[str] = []
    for line in text.splitlines():
        buf: list[str] = []
        i = 0
        in_string = False
        escape = False
        while i < len(line):
            c = line[i]
            if escape:
                buf.append(c)
                escape = False
                i += 1
                continue
            if in_string:
                if c == "\\":
                    escape = True
                    buf.append(c)
                elif c == '"':
                    in_string = False
                    buf.append(c)
                else:
                    buf.append(c)
                i += 1
                continue
            if c == '"':
                in_string = True
                buf.append(c)
                i += 1
                continue
            if c == "/" and i + 1 < len(line) and line[i + 1] == "/":
                break
            if c == "#":
                break
            buf.append(c)
            i += 1
        out_lines.append("".join(buf).rstrip())
    return "\n".join(out_lines)


def loads_json_with_hash_comments(text: str):
    return json.loads(strip_hash_comments(text))
