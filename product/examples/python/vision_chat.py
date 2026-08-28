# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
"""Ask a vision model about a local image.

    pip install openai
    python vision_chat.py --image photo.jpg --prompt "What is in this image?"

Images travel inline as base64 data URIs. oga-server deliberately does not fetch
remote `image_url` values: an inference server that follows URLs supplied by its
clients is an outbound HTTP client, which is a security problem, not a feature.

With no --model, this picks a vision model from /v1/catalog, preferring one that
is already installed on the server so the first run does not trigger a download.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import urllib.request
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - the message is the point
    sys.exit("openai is not installed. Run: pip install openai")


def pick_vision_model(base: str) -> str:
    """Return a vision-capable model id from the server's catalog."""
    with urllib.request.urlopen(f"{base}/v1/catalog", timeout=30) as response:
        catalog = json.load(response)

    vlms = [m for m in catalog.get("models", []) if m.get("capabilities", {}).get("vision")]
    if not vlms:
        sys.exit("the server's catalog has no vision models")

    installed = [m for m in vlms if m.get("installed")]
    return (installed or vlms)[0]["id"]


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:11434", help="oga-server address")
    ap.add_argument("--model", default=None, help="model id (default: pick from the catalog)")
    ap.add_argument("--image", required=True, type=Path, help="image file to send")
    ap.add_argument("--prompt", default="Describe this image in detail.")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    if not args.image.is_file():
        return _fail(f"no such image: {args.image}")

    base = args.base.rstrip("/")
    model = args.model or pick_vision_model(base)
    if args.model is None:
        print(f"# no --model given, using {model}", file=sys.stderr)

    client = OpenAI(base_url=f"{base}/v1", api_key="not-needed")

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": data_uri(args.image)}},
                ],
            }
        ],
        max_tokens=args.max_tokens,
        stream=True,
    )

    usage = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        if chunk.usage:
            usage = chunk.usage
    print()

    if usage is not None:
        # An image is prefill work, so TTFT is the number an image moves, not tok/s.
        ttft = getattr(usage, "ttft_s", None)
        tps = getattr(usage, "tps", None)
        speed = f"TTFT {ttft:.3f}s, {tps:.1f} tok/s" if ttft and tps else "no timing reported"
        print(
            f"# {usage.prompt_tokens} prompt tokens including the image, {speed}",
            file=sys.stderr,
        )
    return 0


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
