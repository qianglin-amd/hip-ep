# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Tool calling over models that were never told about tools.

onnxruntime-genai hands back text. It does not know what a tool is, and neither does
the artifact, so the whole convention has to live here: the tool list is described to
the model in its prompt, and the reply is read back for the marker the model was asked
to use.

Every current instruction-tuned family has been trained on one of a small number of
shapes, and they disagree only in the wrapper. Rather than guess which one a given
artifact learned, the parser accepts all of them. Accepting a shape the model was not
trained on costs nothing; rejecting the one it emits turns a working tool call into
prose, which is the failure this module exists to avoid and the one that is hardest to
diagnose from the outside.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One call the model asked for, in the shape the OpenAI API returns it."""

    id: str
    name: str
    arguments: str  # JSON text, because that is what the wire format carries

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }

    def to_ollama(self) -> dict[str, Any]:
        try:
            parsed = json.loads(self.arguments)
        except ValueError:
            parsed = {}
        return {"function": {"name": self.name, "arguments": parsed}}


def new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


# `<tool_call>` is the ChatML convention used by the Qwen families; Llama 3.1 marks a
# call with `<|python_tag|>`. Both wrap the same JSON object.
_TAGGED = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)
_PYTHON_TAG = re.compile(r"<\|python_tag\|>\s*(?P<body>.*?)(?=<\||$)", re.DOTALL)
_FENCED = re.compile(r"```(?:json|tool_call)?\s*(?P<body>\{.*?\})\s*```", re.DOTALL)


def describe_tools(tools: list[dict[str, Any]]) -> str:
    """The system text that tells a model which tools exist and how to call one.

    The schema is passed through as JSON rather than prose because that is how these
    models saw it in training, and because a hand-written summary would silently drop
    the constraints a caller put in the schema.
    """
    described = []
    for tool in tools:
        function = tool.get("function") if tool.get("type") == "function" else tool
        if not isinstance(function, dict) or not function.get("name"):
            continue
        described.append(
            {
                "name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object"}),
            }
        )
    if not described:
        return ""

    listing = "\n".join(json.dumps(entry, ensure_ascii=False) for entry in described)
    return (
        "You have access to the following tools:\n"
        f"{listing}\n\n"
        "When a tool is needed, reply with only a tool call, in this exact form:\n"
        '<tool_call>{"name": "<tool name>", "arguments": {<arguments>}}</tool_call>\n'
        "Call a tool only when it is needed to answer, and use the exact tool name. "
        "Otherwise answer normally."
    )


def _coerce(body: str) -> ToolCall | None:
    """Read a call out of a wrapper body, tolerating debris around the object.

    A model that has been asked for a tool call does not always produce a clean
    object inside the wrapper. Llama 3.1 emits the tool's name again first, as
    `<tool_call>get_weather>{"name": ...}</tool_call>`, which no JSON parser accepts.
    Refusing that loses a call the model clearly made, so the first balanced object
    inside the body is tried before giving up.
    """
    call = _from_json(body)
    if call is not None:
        return call
    for _, _, span in _balanced_objects(body):
        call = _from_json(span)
        if call is not None:
            return call
    return None


def _from_json(body: str) -> ToolCall | None:
    """Read one candidate JSON object into a call, or reject it."""
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None

    name = parsed.get("name") or parsed.get("tool") or parsed.get("function")
    if isinstance(name, dict):  # {"function": {"name": ...}}
        name = name.get("name")
    if not isinstance(name, str) or not name:
        return None

    # Families disagree on the key: "arguments" in ChatML, "parameters" in Llama's.
    arguments = parsed.get("arguments")
    if arguments is None:
        arguments = parsed.get("parameters")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        # Already serialised; keep it if it is JSON, otherwise wrap it so the caller
        # always receives parseable arguments.
        try:
            json.loads(arguments)
            text = arguments
        except ValueError:
            text = json.dumps({"input": arguments}, ensure_ascii=False)
    else:
        text = json.dumps(arguments, ensure_ascii=False)

    return ToolCall(id=new_call_id(), name=name, arguments=text)


def _balanced_objects(text: str) -> list[tuple[int, int, str]]:
    """Every top-level ``{...}`` span in `text`, ignoring braces inside strings."""
    spans: list[tuple[int, int, str]] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append((start, index + 1, text[start : index + 1]))
    return spans


def extract_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split a completion into the prose the user should see and the calls to run.

    Returns the text with any call markup removed, so a model that narrates before
    calling does not have its markup shown to the user.
    """
    if not text:
        return text, []

    # `<tool_call>` and `<|python_tag|>` mean a tool call and nothing else, so they are
    # removed whether or not their contents parsed, and a model that emits one of each
    # has both removed. Stripping only the wrapper that happened to parse leaves the
    # other one on screen, which is what a real model did: it marked its call with
    # `<|python_tag|>` and left a malformed `<tool_call>` fragment behind.
    tagged: list[ToolCall] = []
    for pattern in (_TAGGED, _PYTHON_TAG):
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        tagged.extend(call for m in matches if (call := _coerce(m.group("body").strip())))
        text = pattern.sub("", text)
    if tagged:
        return text.strip(), tagged

    # A fenced block is ordinary markdown, so unlike the wrappers above it is removed
    # only when it really did carry a call.
    fenced_matches = list(_FENCED.finditer(text))
    fenced = [call for m in fenced_matches if (call := _coerce(m.group("body").strip()))]
    if fenced:
        return _FENCED.sub("", text).strip(), fenced

    # No wrapper at all. A bare object is only treated as a call when it names a tool
    # and carries an argument key, so ordinary JSON in an answer is left alone.
    spans = _balanced_objects(text)
    calls = []
    consumed: list[tuple[int, int]] = []
    for start, end, body in spans:
        parsed_ok = _coerce(body)
        if parsed_ok is None:
            continue
        try:
            raw = json.loads(body)
        except ValueError:
            continue
        if "arguments" not in raw and "parameters" not in raw:
            continue
        calls.append(parsed_ok)
        consumed.append((start, end))

    if not calls:
        return text, []

    cleaned = text
    for start, end in reversed(consumed):
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned.strip(), calls
