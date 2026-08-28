# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Stream a chat completion from oga-server using the official OpenAI SDK.

oga-server implements the OpenAI chat API, so the only change an existing
application needs is `base_url`. The API key is unused but the SDK requires a
non-empty string.

    pip install openai
    python openai_chat.py --prompt "Explain what an execution provider does."

Beyond the standard response, HIP EP adds `ttft_s`, `decode_s` and `tps` to the
`usage` object, which is how the console and the benchmark harness report speed.
The SDK keeps unknown fields, so they are readable as attributes.
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - the message is the point
    sys.exit("openai is not installed. Run: pip install openai")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:11434", help="oga-server address")
    ap.add_argument("--model", default=None, help="model id (default: first model listed)")
    ap.add_argument("--prompt", default="Explain what an execution provider does.")
    ap.add_argument("--system", default=None, help="optional system prompt")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-stream", action="store_true", help="wait for the full response")
    args = ap.parse_args()

    client = OpenAI(base_url=f"{args.base.rstrip('/')}/v1", api_key="not-needed")

    model = args.model
    if model is None:
        available = client.models.list().data
        if not available:
            return _fail("the server reports no models. Install one from the model zoo first.")
        model = available[0].id
        print(f"# no --model given, using {model}", file=sys.stderr)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})

    request = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    started = time.perf_counter()

    if args.no_stream:
        response = client.chat.completions.create(**request)
        print(response.choices[0].message.content)
        report(response.usage, time.perf_counter() - started)
        return 0

    usage = None
    first_token_at = None
    for chunk in client.chat.completions.create(**request, stream=True):
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(chunk.choices[0].delta.content, end="", flush=True)
        # The final frame carries no delta, only the finish reason and usage.
        if chunk.usage:
            usage = chunk.usage
    print()

    if first_token_at is not None:
        print(f"\n# client-observed TTFT {first_token_at - started:.3f}s", file=sys.stderr)
    report(usage, time.perf_counter() - started)
    return 0


def report(usage, wall_s: float) -> None:
    """Print the server-reported speed alongside the wall time we measured."""
    if usage is None:
        print(f"# {wall_s:.2f}s total", file=sys.stderr)
        return

    # ttft_s / decode_s / tps are HIP EP extensions; getattr keeps this example
    # working against a stock OpenAI endpoint too.
    ttft = getattr(usage, "ttft_s", None)
    tps = getattr(usage, "tps", None)
    parts = [f"{usage.completion_tokens} tokens", f"{wall_s:.2f}s total"]
    if ttft is not None:
        parts.append(f"TTFT {ttft:.3f}s")
    if tps is not None:
        parts.append(f"{tps:.1f} tok/s")
    print("# " + ", ".join(parts), file=sys.stderr)


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
