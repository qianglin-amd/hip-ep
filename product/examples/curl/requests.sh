#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Everything the console does is reachable from a shell. Run the whole file as a tour of
# the API, or copy the block you need.
#
#   ./requests.sh [base-url] [model-id]
#
# Requires curl and jq. On Windows, run it from Git Bash or WSL.

set -euo pipefail

BASE="${1:-http://127.0.0.1:11434}"
MODEL="${2:-}"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required: https://jqlang.github.io/jq/download/" >&2
  exit 1
fi

if [[ -z "${MODEL}" ]]; then
  MODEL=$(curl -fsS "${BASE}/v1/models" | jq -r '.data[0].id')
  echo "# no model given, using ${MODEL}" >&2
fi

echo "## health and engine"
curl -fsS "${BASE}/health" | jq .

echo "## certified catalog, joined with what is installed on this machine"
curl -fsS "${BASE}/v1/catalog" |
  jq '.models[] | select(.tier == "llm") | {id, quantization: .quant_label, installed, fits}'

echo "## one completion"
curl -fsS "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "${MODEL}" '{
        model: $m,
        messages: [{role: "user", content: "Name one advantage of INT4 quantization."}],
        max_tokens: 120
      }')" |
  jq '{content: .choices[0].message.content, usage}'

echo "## the same completion, streamed"
# -N disables curl's buffering; without it the deltas arrive in one block at the end.
curl -fsS -N "${BASE}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "${MODEL}" '{
        model: $m,
        messages: [{role: "user", content: "Count to five."}],
        max_tokens: 60,
        stream: true
      }')" |
  while read -r line; do
    [[ "${line}" == data:* ]] || continue
    payload="${line#data: }"
    [[ "${payload}" == '"[DONE]"' || "${payload}" == "[DONE]" ]] && break
    printf '%s' "$(echo "${payload}" | jq -r '.choices[0].delta.content // ""')"
  done
echo

echo "## download a certified model, following the job to completion"
JOB=$(curl -fsS "${BASE}/v1/pull" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "${MODEL}" '{model: $m}')" | jq -r '.job.id')
echo "job ${JOB}"
# The event stream ends when the job does, so this blocks until the pull finishes.
curl -fsS -N "${BASE}/v1/jobs/${JOB}/events" | grep --line-buffered -o '"stage":"[^"]*"' || true

echo "## benchmark this machine and compare against the certified numbers"
BENCH=$(curl -fsS "${BASE}/v1/benchmarks/run" \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg m "${MODEL}" '{
        model: $m, contexts: [128, 2048], generate_tokens: 64, repeats: 3, warmup: 1
      }')" | jq -r '.job.id')
curl -fsS -N "${BASE}/v1/jobs/${BENCH}/events" >/dev/null
curl -fsS "${BASE}/v1/jobs/${BENCH}" | jq '.result.contexts, .result.delta_vs_published'

echo "## live server load, including other users (Ctrl-C to stop)"
echo "curl -N ${BASE}/v1/status/events"
