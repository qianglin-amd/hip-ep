// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Live job panel, shared by pull, port and benchmark. Every long operation on the
// server reports the same stage/log/result shape, so one renderer covers all of them.

import { api, subscribe } from "../api.js";
import { copyButton, el, fill, fmtDuration, fmtNumber, timeAgo, toast } from "../ui.js";

const STATE_LABEL = {
  pending: "waiting",
  running: "running",
  succeeded: "done",
  failed: "failed",
  cancelled: "cancelled",
};

/**
 * Build a self-updating panel for one job.
 *
 * The caller gets a DOM node; the panel owns its own SSE subscription and closes it
 * when the job reaches a terminal state or when the node leaves the document.
 */
export function jobPanel(jobId, { onFinish, showResult = true } = {}) {
  const header = el("div", { class: "job-head" });
  const stages = el("div", { class: "stages" });
  const logBox = el("pre", { class: "log" });
  const resultBox = el("div", { class: "job-result", hidden: true });

  const root = el("div", { class: "job-panel" }, [
    header,
    stages,
    resultBox,
    el("details", { class: "log-wrap" }, [
      el("summary", { text: "Log" }),
      logBox,
    ]),
  ]);

  let stream = null;
  let finished = false;
  let logCount = 0;

  function paintHeader(job) {
    const cancellable = job.state === "running" || job.state === "pending";
    fill(header, 
      el("div", { class: "row" }, [
        el("span", { class: `badge badge-${job.state}` }, [
          el("span", { class: "dot" }),
          el("span", { text: STATE_LABEL[job.state] || job.state }),
        ]),
        el("span", { class: "mono faint", text: job.id }),
        el("span", { class: "grow" }),
        job.duration_s !== null && job.duration_s !== undefined
          ? el("span", { class: "faint", text: fmtDuration(job.duration_s) })
          : null,
        cancellable
          ? el("button", {
              class: "btn btn-sm btn-danger",
              text: "Cancel",
              onclick: async (event) => {
                event.currentTarget.disabled = true;
                try {
                  await api.cancelJob(job.id);
                } catch (error) {
                  toast(error.message, "error");
                }
              },
            })
          : null,
      ]),
      el("div", { class: "bar bar-lg" },
        el("i", {
          class: job.state === "failed" ? "bad" : "",
          style: `width:${(job.progress * 100).toFixed(1)}%`,
        }),
      ),
      job.error ? el("div", { class: "notice bad", text: job.error }) : null,
    );
  }

  function paintStages(job) {
    fill(stages, 
      ...job.stages.map((stage) =>
        el("div", { class: `stage-row ${stage.state}${stage.skipped ? " skipped" : ""}` }, [
          // A running stage is marked by the spinning border in CSS, so it carries no glyph.
          el("span", { class: "mark" }, stage.state === "running"
            ? null
            : el("span", {
                text: stage.state === "succeeded" ? "\u2713"
                  : stage.state === "failed" ? "\u2717"
                  : stage.skipped ? "\u2013" : "",
              }),
          ),
          el("div", { class: "grow" }, [
            el("div", { class: "row" }, [
              el("strong", { text: stage.label }),
              el("span", { class: "grow" }),
              stage.duration_s
                ? el("span", { class: "faint", text: fmtDuration(stage.duration_s) })
                : null,
            ]),
            stage.detail ? el("div", { class: "faint small", text: stage.detail }) : null,
            stage.state === "running"
              ? el("div", { class: "bar" }, el("i", { style: `width:${(stage.progress * 100).toFixed(0)}%` }))
              : null,
          ]),
        ]),
      ),
    );
  }

  function paintLogs(job) {
    const lines = job.logs || [];
    if (lines.length === logCount) return;
    logCount = lines.length;
    const stick = logBox.scrollHeight - logBox.scrollTop - logBox.clientHeight < 40;
    logBox.textContent = lines
      .map((line) => `${new Date(line.at * 1000).toLocaleTimeString()}  ${line.text}`)
      .join("\n");
    if (stick) logBox.scrollTop = logBox.scrollHeight;
  }

  function paintResult(job) {
    if (!showResult || !job.result || !Object.keys(job.result).length) return;
    resultBox.hidden = false;
    const result = job.result;
    const entries = Object.entries(result).filter(
      ([, value]) => value !== null && typeof value !== "object",
    );
    fill(resultBox, 
      el("h4", { text: "Result" }),
      el("div", { class: "kv" },
        entries.flatMap(([key, value]) => [
          el("span", { class: "k", text: key.replace(/_/g, " ") }),
          el("span", { class: "v mono", text: String(value) }),
        ]),
      ),
      result.share_command
        ? el("div", { class: "share" }, [
            el("code", { text: result.share_command }),
            copyButton(() => result.share_command, "Copy"),
          ])
        : null,
      result.summary ? benchSummary(result.summary) : null,
    );
  }

  function paint(job) {
    paintHeader(job);
    paintStages(job);
    paintLogs(job);
    paintResult(job);

    if (!finished && ["succeeded", "failed", "cancelled"].includes(job.state)) {
      finished = true;
      stream?.close();
      onFinish?.(job);
    }
  }

  // The SSE stream carries summaries without logs, so fetch the full snapshot on
  // every update tick would be wasteful; instead pull logs once at the end and
  // whenever the stage set advances.
  let lastStage = null;
  async function refreshDetail() {
    try {
      paint(await api.job(jobId));
    } catch (error) {
      fill(root, el("div", { class: "notice bad", text: error.message }));
    }
  }

  refreshDetail();
  stream = subscribe(`/v1/jobs/${jobId}/events`, {
    onMessage: (job) => {
      if (typeof job !== "object") return;
      paint({ ...job, logs: job.logs });
      if (job.current_stage !== lastStage || job.state !== "running") {
        lastStage = job.current_stage;
      }
    },
    onError: () => refreshDetail(),
  });

  root.close = () => stream?.close();
  return root;
}

/**
 * Render the comparison table a benchmark job produces.
 *
 * TTFT and tokens/s are the engine's own measurements, which is what the certified
 * numbers are, so the two are comparable. The extra latency an HTTP client sees on top
 * of that is shown as a separate line rather than folded into the comparison.
 *
 * Sign convention: a faster TTFT is a negative percentage on the wire, so it is
 * flipped here to keep "green means better" true for both columns.
 */
export function benchSummary(summary) {
  const contexts = Object.entries(summary.contexts || {});
  if (!contexts.length) return null;
  const deltas = summary.delta_vs_published || {};
  const published = summary.published || {};

  const deltaCell = (pct, betterWhenLower = false) => {
    if (pct === null || pct === undefined) return el("td", { class: "num faint", text: "-" });
    const good = betterWhenLower ? pct <= 0 : pct >= 0;
    return el("td", {
      class: `num ${good ? "delta-up" : "delta-down"}`,
      text: `${pct >= 0 ? "+" : ""}${fmtNumber(pct, 1)}%`,
    });
  };

  return el("div", {}, [
    el("table", { class: "data", style: "margin-top:12px" }, [
      el("thead", {}, el("tr", {}, [
        el("th", { text: "Prompt tokens" }),
        el("th", { class: "right", text: "TTFT" }),
        el("th", { class: "right", text: "vs certified" }),
        el("th", { class: "right", text: "Tokens/s" }),
        el("th", { class: "right", text: "vs certified" }),
        el("th", { class: "right", text: "Runs" }),
      ])),
      el("tbody", {}, contexts.map(([context, row]) => {
        const delta = deltas[context] || {};
        const ref = published[context] || {};
        return el("tr", {}, [
          el("td", { class: "mono", text: Number(context).toLocaleString() }),
          el("td", { class: "num" }, [
            el("span", { text: `${fmtNumber(row.ttft_s, 3)} s` }),
            ref.ttft_s
              ? el("div", { class: "faint small", text: `certified ${fmtNumber(ref.ttft_s, 2)} s` })
              : null,
            row.e2e_ttft_s
              ? el("div", { class: "faint small", text: `${fmtNumber(row.e2e_ttft_s, 3)} s end to end` })
              : null,
          ]),
          deltaCell(delta.ttft_pct, true),
          el("td", { class: "num" }, [
            el("span", { text: fmtNumber(row.tps) }),
            ref.tps ? el("div", { class: "faint small", text: `certified ${fmtNumber(ref.tps)}` }) : null,
          ]),
          deltaCell(delta.tps_pct),
          el("td", { class: "num faint", text: String(row.runs) }),
        ]);
      })),
    ]),
    el("div", { class: "faint small", style: "margin-top:8px" },
      `${summary.harness || ""} · ${summary.generate_tokens} new tokens · ` +
      `${summary.repeats} repeats after ${summary.warmup} warmup run(s)`),
  ]);
}

/** Compact one-line job row for lists. */
export function jobRow(job, { onOpen } = {}) {
  return el("tr", { class: "clickable", onclick: () => onOpen?.(job) }, [
    el("td", {}, [
      el("div", { text: job.title }),
      el("div", { class: "faint small mono", text: job.id }),
    ]),
    el("td", {}, el("span", { class: `badge badge-${job.state}` }, [
      el("span", { class: "dot" }),
      el("span", { text: STATE_LABEL[job.state] || job.state }),
    ])),
    el("td", { style: "width:180px" },
      el("div", { class: "bar" }, el("i", {
        class: job.state === "failed" ? "bad" : "",
        style: `width:${(job.progress * 100).toFixed(0)}%`,
      })),
    ),
    el("td", { class: "num faint", text: timeAgo(job.created_at) }),
  ]);
}
