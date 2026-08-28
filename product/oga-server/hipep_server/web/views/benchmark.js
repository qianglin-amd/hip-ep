// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Benchmarks: the certified table shipped with the repo, plus runs measured here.

import { api } from "../api.js";
import { state } from "../app.js";
import { clear, el, empty, fill, fmtNumber, modal, timeAgo, toast } from "../ui.js";
import { benchSummary, jobPanel } from "./jobview.js";

const CONTEXTS = [128, 2048, 16384];

export async function renderBenchmark(root) {
  const tabs = el("div", { class: "tabs" });
  const body = el("div", {});
  root.append(tabs, body);

  const panels = {
    published: { label: "Certified results", render: paintPublished },
    run: { label: "Run on this machine", render: paintRun },
    local: { label: "Local results", render: paintLocal },
  };

  let openPanel = null;

  for (const [key, panel] of Object.entries(panels)) {
    tabs.append(
      el("button", {
        class: key === "published" ? "active" : "",
        onclick: (event) => {
          for (const b of tabs.children) b.classList.remove("active");
          event.currentTarget.classList.add("active");
          clear(body);
          panel.render(body);
        },
      }, panel.label),
    );
  }

  await paintPublished(body);

  /* ---------- certified ---------- */

  async function paintPublished(host) {
    let data;
    try {
      data = await api.published();
    } catch (error) {
      host.append(el("div", { class: "notice bad", text: error.message }));
      return;
    }

    const meta = data.meta || {};
    const generative = data.results.filter((r) => r.tier === "llm" || r.tier === "vlm");
    const speech = data.results.filter((r) => r.tier === "speech");
    const vision = data.results.filter((r) => r.tier === "vision");

    host.append(
      el("div", { class: "card notice-card" }, [
        el("div", { class: "row wrap" }, [
          el("div", {}, [
            el("h3", { text: `Release ${meta.release || "-"}` }),
            el("div", { class: "muted small" },
              `Measured ${meta.run_date || "-"} on ${meta.device?.name || "the certification host"}` +
              `${meta.device?.gfx_arch ? ` (${meta.device.gfx_arch}, ${meta.device.memory_gb} GB)` : ""}`),
          ]),
          el("span", { class: "grow" }),
          ...Object.entries(meta.pass_rate || {}).map(([suite, rate]) =>
            el("div", { class: "stat" }, [
              el("div", { class: "k", text: suite }),
              el("div", { class: "v", text: `${fmtNumber(rate * 100, 1)}%` }),
            ]),
          ),
        ]),
        el("div", { class: "faint small", style: "margin-top:10px" },
          "These are the numbers the model zoo is certified against. Run the same " +
          "workload locally from the next tab to compare your machine."),
      ]),
    );

    if (generative.length) {
      host.append(
        el("h3", { style: "margin-top:24px", text: "LLM and VLM" }),
        el("div", { class: "table-wrap" }, generativeTable(generative)),
      );
    }
    if (speech.length) {
      host.append(
        el("h3", { style: "margin-top:24px", text: "Speech" }),
        el("div", { class: "table-wrap" },
          el("table", { class: "data" }, [
            el("thead", {}, el("tr", {}, [
              el("th", { text: "Model" }),
              el("th", { class: "right", text: "TTFT" }),
              el("th", { class: "right", text: "Tokens/s" }),
              el("th", { class: "right", text: "Real-time factor" }),
            ])),
            el("tbody", {}, speech.map((row) =>
              el("tr", {}, [
                el("td", {}, [
                  el("div", { text: row.name }),
                  el("div", { class: "faint small mono", text: row.model }),
                ]),
                el("td", { class: "num", text: num(row.ttft_s?.["0"], 3, " s") }),
                el("td", { class: "num", text: num(row.tps?.["0"]) }),
                el("td", { class: "num", text: row.rtf ? `${fmtNumber(row.rtf, 1)}x` : "-" }),
              ]),
            )),
          ]),
        ),
      );
    }
    if (vision.length) {
      host.append(
        el("h3", { style: "margin-top:24px", text: "Vision" }),
        el("div", { class: "table-wrap" },
          el("table", { class: "data" }, [
            el("thead", {}, el("tr", {}, [
              el("th", { text: "Model" }),
              el("th", { class: "right", text: "First inference" }),
              el("th", { class: "right", text: "Average" }),
              el("th", { class: "right", text: "Queries/s" }),
              el("th", { text: "Status" }),
            ])),
            el("tbody", {}, vision.map((row) =>
              el("tr", { class: row.status === "failing" ? "row-bad" : "" }, [
                el("td", {}, [
                  el("div", { text: row.name }),
                  el("div", { class: "faint small mono", text: row.model }),
                ]),
                el("td", { class: "num", text: num(row.first_ms, 1, " ms") }),
                el("td", { class: "num", text: num(row.avg_ms, 2, " ms") }),
                el("td", { class: "num", text: num(row.qps, 1) }),
                el("td", {},
                  el("span", {
                    class: `tag ${row.status === "failing" ? "tag-bad" : "tag-ok"}`,
                    text: row.status,
                  }),
                ),
              ]),
            )),
          ]),
        ),
      );
    }
  }

  function num(value, digits = 1, suffix = "") {
    return value === null || value === undefined ? "-" : `${fmtNumber(value, digits)}${suffix}`;
  }

  function generativeTable(rows) {
    const header = el("tr", {}, [
      el("th", { rowspan: "2", text: "Model" }),
      el("th", { rowspan: "2", text: "Quantization" }),
      ...CONTEXTS.map((ctx) =>
        el("th", { colspan: "2", class: "center group", text: `${ctx.toLocaleString()} tokens` }),
      ),
    ]);
    const subheader = el("tr", {}, CONTEXTS.flatMap(() => [
      el("th", { class: "right small", text: "TTFT" }),
      el("th", { class: "right small", text: "tok/s" }),
    ]));

    return el("table", { class: "data" }, [
      el("thead", {}, [header, subheader]),
      el("tbody", {}, rows.map((row) =>
        el("tr", {}, [
          el("td", {}, [
            el("div", { class: "row", style: "gap:6px" }, [
              el("span", { text: row.name }),
              el("span", { class: `tag tag-${row.tier}`, text: row.tier.toUpperCase() }),
            ]),
            el("div", { class: "faint small mono", text: row.model }),
          ]),
          el("td", { class: "small", text: row.quantization }),
          ...CONTEXTS.flatMap((ctx) => {
            const key = String(ctx);
            const ttft = row.ttft_s?.[key];
            const tps = row.tps?.[key];
            const prevTps = row.previous?.tps?.[key];
            const cells = [el("td", { class: "num", text: num(ttft, 2, " s") })];
            const tpsCell = el("td", { class: "num" }, [el("span", { text: num(tps) })]);
            if (prevTps && tps) {
              const pct = ((tps - prevTps) / prevTps) * 100;
              if (Math.abs(pct) >= 1) {
                tpsCell.append(
                  el("div", {
                    class: `small ${pct >= 0 ? "delta-up" : "delta-down"}`,
                    text: `${pct >= 0 ? "+" : ""}${pct.toFixed(0)}%`,
                  }),
                );
              }
            }
            cells.push(tpsCell);
            return cells;
          }),
        ]),
      )),
    ]);
  }

  /* ---------- run locally ---------- */

  async function paintRun(host) {
    const models = state.catalog.filter((m) => m.tier === "llm" || m.tier === "vlm");
    const modelSelect = el("select", {}, models.map((m) =>
      el("option", { value: m.id }, `${m.name}${m.installed ? "" : "  (not downloaded)"}`),
    ));
    const contextChecks = CONTEXTS.map((ctx) =>
      el("label", { class: "check" }, [
        el("input", { type: "checkbox", value: String(ctx), checked: ctx !== 16384 }),
        `${ctx.toLocaleString()} tokens`,
      ]),
    );
    const tokensInput = el("input", { type: "number", min: "1", value: "64" });
    const repeatsInput = el("input", { type: "number", min: "1", max: "20", value: "3" });
    const warmupInput = el("input", { type: "number", min: "0", max: "5", value: "1" });
    const panelHost = el("div", {});

    const start = el("button", { class: "btn btn-primary", text: "Run benchmark" });

    host.append(
      el("div", { class: "grid grid-2 grid-wide" }, [
        el("div", { class: "card" }, [
          el("h3", { text: "Workload" }),
          el("label", { class: "field" }, ["Model", modelSelect]),
          el("label", { class: "field" }, ["Prompt lengths", el("div", { class: "col" }, contextChecks)]),
          el("div", { class: "grid grid-3" }, [
            el("label", { class: "field" }, ["New tokens", tokensInput]),
            el("label", { class: "field" }, ["Repeats", repeatsInput]),
            el("label", { class: "field" }, ["Warmup", warmupInput]),
          ]),
          el("div", { class: "faint small" },
            "Measured through this server's /v1/chat/completions, so the numbers include " +
            "the same overhead your application will see. Warmup runs are discarded, which " +
            "keeps first-run kernel compilation out of the result."),
          el("div", { class: "row", style: "margin-top:16px" }, [start]),
        ]),
        el("div", { class: "card" }, [el("h3", { text: "Progress" }), panelHost]),
      ]),
    );

    start.addEventListener("click", async () => {
      const contexts = contextChecks
        .filter((label) => label.querySelector("input").checked)
        .map((label) => Number(label.querySelector("input").value));
      if (!contexts.length) {
        toast("Pick at least one prompt length", "error");
        return;
      }
      start.disabled = true;
      try {
        const { job } = await api.runBenchmark({
          model: modelSelect.value,
          contexts,
          generate_tokens: Number(tokensInput.value) || 64,
          repeats: Number(repeatsInput.value) || 3,
          warmup: Number(warmupInput.value) || 0,
        });
        openPanel?.close?.();
        openPanel = jobPanel(job.id, {
          onFinish: (finished) => {
            if (finished.state === "succeeded") toast("Benchmark complete", "ok");
            start.disabled = false;
          },
        });
        fill(panelHost, openPanel);
      } catch (error) {
        toast(error.message, "error", 8000);
        start.disabled = false;
      }
    });
  }

  /* ---------- local history ---------- */

  async function paintLocal(host) {
    let runs = [];
    try {
      runs = (await api.localBenchmarks()).runs;
    } catch (error) {
      host.append(el("div", { class: "notice bad", text: error.message }));
      return;
    }
    if (!runs.length) {
      host.append(empty(
        "No local runs yet",
        "Run a benchmark from the previous tab. Results are written to the data directory and listed here.",
      ));
      return;
    }
    host.append(
      el("div", { class: "table-wrap" },
        el("table", { class: "data" }, [
          el("thead", {}, el("tr", {}, [
            el("th", { text: "Model" }),
            el("th", { text: "When" }),
            el("th", { class: "right", text: "Prompt lengths" }),
            el("th", { class: "right", text: "Best tok/s" }),
            el("th", {}),
          ])),
          el("tbody", {}, runs.map((run) => {
            const contexts = Object.entries(run.contexts || {});
            const best = Math.max(0, ...contexts.map(([, v]) => v.tps || 0));
            return el("tr", {}, [
              el("td", {}, [
                el("div", { text: run.model_name || run.model }),
                el("div", { class: "faint small mono", text: run.quantization || "" }),
              ]),
              el("td", { class: "faint", text: timeAgo(run.finished_at) }),
              el("td", { class: "num mono", text: contexts.map(([k]) => k).join(", ") }),
              el("td", { class: "num", text: fmtNumber(best) }),
              el("td", { class: "right" },
                el("button", {
                  class: "btn btn-sm",
                  text: "Details",
                  onclick: () => modal(run.model_name || run.model, benchSummary(run), {
                    subtitle: `measured ${new Date(run.finished_at * 1000).toLocaleString()}`,
                  }),
                }),
              ),
            ]);
          })),
        ]),
      ),
    );
  }

  return () => openPanel?.close?.();
}
