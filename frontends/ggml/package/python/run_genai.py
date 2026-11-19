#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
"""Chat with an ONNX Runtime GenAI model (a directory with genai_config.json)
on the HIP EP.

    python run_genai.py <model_dir> [-p "prompt"] [--max-length N]

Without -p it starts an interactive chat; an empty line exits.
"""

import argparse
import json
import os
import time

import onnxruntime_genai as og

import hipep


def generate(model, tokenizer, prompt, max_length, template=None):
    messages = json.dumps([{"role": "user", "content": prompt}])
    try:
        text = tokenizer.apply_chat_template(
            messages=messages, template_str=template, add_generation_prompt=True
        )
    except Exception:
        # Models without a chat template take the raw prompt.
        text = prompt
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=max_length, do_sample=False)
    generator = og.Generator(model, params)
    generator.append_tokens(tokenizer.encode(text))
    stream = tokenizer.create_stream()
    count = 0
    start = time.perf_counter()
    while not generator.is_done():
        generator.generate_next_token()
        print(stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
        count += 1
    elapsed = time.perf_counter() - start
    print(f"\n[{count} tokens, {count / elapsed:.1f} tokens/s]")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model_dir")
    ap.add_argument("-p", "--prompt")
    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="prompt plus reply tokens (default 1024)",
    )
    args = ap.parse_args()

    start = time.perf_counter()
    model = og.Model(hipep.genai_config(args.model_dir))
    tokenizer = og.Tokenizer(model)
    print(
        f"model loaded on {hipep.GENAI_EP_NAME} in {time.perf_counter() - start:.1f} s"
    )

    # Some exports keep the chat template in its own file rather than in
    # tokenizer_config.json.
    template = None
    template_file = os.path.join(args.model_dir, "chat_template.jinja")
    if os.path.isfile(template_file):
        with open(template_file, encoding="utf-8") as f:
            template = f.read()

    if args.prompt:
        generate(model, tokenizer, args.prompt, args.max_length, template)
        return 0
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            break
        if not prompt:
            break
        generate(model, tokenizer, prompt, args.max_length, template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
