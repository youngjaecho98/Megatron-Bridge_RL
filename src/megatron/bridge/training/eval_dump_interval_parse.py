# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build ``intervals`` dicts for eval_dump / supervised_span_text (stdlib only)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _normalize_span(span: str) -> str:
    s = (span or "").strip()
    if s.startswith("<th"):
        s = "[\n" + s[3:]
    return s


def _try_json_interval_objects(s: str) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    if not s:
        return [], "empty", None

    candidates = [s]
    lb, rb = s.find("["), s.rfind("]")
    if lb != -1 and rb != -1 and rb > lb:
        inner = s[lb : rb + 1].strip()
        if inner and inner != s:
            candidates.append(inner)

    last_err: Optional[str] = None
    for cand in candidates:
        try:
            data = json.loads(cand)
        except json.JSONDecodeError as e:
            last_err = str(e)
            continue
        if isinstance(data, dict):
            return [data], "ok", None
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)], "ok", None

    return [], "json_error", last_err


def _read_json_string(s: str, value_start: int) -> Tuple[str, int]:
    i = value_start
    parts: List[str] = []
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s):
            parts.append(s[i + 1])
            i += 2
            continue
        if ch == '"':
            return "".join(parts), i + 1
        parts.append(ch)
        i += 1
    return "".join(parts), i


def _extract_string_field(chunk: str, key: str) -> str:
    pat = re.compile(rf'"{re.escape(key)}"\s*:\s*"')
    m = pat.search(chunk)
    if not m:
        return ""
    text, _ = _read_json_string(chunk, m.end())
    return text


def _brace_balanced_slice(s: str, open_idx: int) -> Optional[Tuple[str, int]]:
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    in_str = False
    esc = False
    i = open_idx
    while i < len(s):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx : i + 1], i + 1
        i += 1
    return None


def _extract_guardrail(chunk: str) -> Any:
    m = re.search(r'"GUARDRAIL"\s*:\s*\{', chunk)
    if not m:
        return {}
    braced = _brace_balanced_slice(chunk, m.end() - 1)
    if not braced:
        return {}
    blob, _ = braced
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {"_unparsed": blob}


def _loose_parse_one_chunk(chunk: str) -> Optional[Dict[str, Any]]:
    ms = re.search(r'"START_SEC"\s*:\s*([-0-9.eE+]+)', chunk)
    me = re.search(r'"END_SEC"\s*:\s*([-0-9.eE+]+)', chunk)
    if not ms or not me:
        return None
    try:
        start = float(ms.group(1))
        end = float(me.group(1))
    except ValueError:
        return None
    desc = _extract_string_field(chunk, "DESCRIPTION")
    expl = _extract_string_field(chunk, "EXPLANATION")
    guard = _extract_guardrail(chunk)
    return {
        "START_SEC": start,
        "END_SEC": end,
        "DESCRIPTION": desc,
        "GUARDRAIL": guard,
        "EXPLANATION": expl,
    }


def _loose_interval_objects(s: str) -> List[Dict[str, Any]]:
    markers = [m.start() for m in re.finditer(r'"START_SEC"\s*:', s)]
    if not markers:
        return []
    out: List[Dict[str, Any]] = []
    for i, pos in enumerate(markers):
        end = markers[i + 1] if i + 1 < len(markers) else len(s)
        chunk = s[pos:end]
        one = _loose_parse_one_chunk(chunk)
        if one:
            out.append(one)
    return out


def intervals_presentation_for_span(span_text: str) -> Dict[str, Any]:
    """JSON parse first; on failure use loose key/regex scan. Same keys as eval_dump ``intervals``."""

    s = _normalize_span(span_text)
    objs, status, err = _try_json_interval_objects(s)

    if status == "empty":
        objs = []
        err = None
        parse_status = "empty"
        parse_note = "supervised_span_text is blank."
    elif status == "ok" and not objs:
        objs = []
        err = None
        parse_status = "empty"
        parse_note = "Parsed as JSON with zero interval objects (e.g. `[]` — no violations)."
    elif status == "ok" and objs:
        parse_status = "ok"
        parse_note = None
    else:
        loose = _loose_interval_objects(s)
        if loose:
            objs = loose
            parse_status = "loose"
            parse_note = (
                "Full JSON parse failed; filled fields via key/regex scan. "
                "Values may be truncated or mis-aligned vs strict JSON."
            )
        else:
            parse_status = "json_error" if s else "empty"
            parse_note = None

    out: Dict[str, Any] = {
        "parse_status": parse_status,
        "parse_ok": parse_status in ("ok", "loose", "empty"),
        "chunks": {},
        "chunk_order": [],
    }
    if err and parse_status == "json_error":
        out["parse_error"] = err
    if parse_note:
        out["parse_note"] = parse_note
    if parse_status == "loose":
        out["parse_loose"] = True

    for i, obj in enumerate(objs):
        cid = i + 1
        key = f"chunk_{cid}"
        out["chunk_order"].append(key)
        out["chunks"][key] = {
            "start": obj.get("START_SEC"),
            "end": obj.get("END_SEC"),
            "description": obj.get("DESCRIPTION"),
            "guardrail": obj.get("GUARDRAIL") if isinstance(obj.get("GUARDRAIL"), dict) else obj.get("GUARDRAIL"),
            "explanation": obj.get("EXPLANATION"),
        }
    if parse_status == "json_error" and not objs:
        out["parse_hint"] = (
            "Neither JSON nor loose scan produced intervals; span may lack recognizable "
            '"START_SEC" fields.'
        )
    return out
