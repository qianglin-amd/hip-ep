// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Model zoo: the certified catalog joined with what is installed locally and whether
// each model fits on this machine.

import { api } from "../api.js";
import { onState, refreshCatalog, state } from "../app.js";
import {
  clear, el, empty, fmtBytes, fmtContext, fmtNumber, modal, toast,
} from "../ui.js";
import { jobPanel } from "./jobview.js";

const TIER_LABEL = { llm: "LLM", vlm: "VLM", speech: "Speech", vision: "Vision" };

export async function renderModels(root) {
  let tier = "llm";
  let query = "";
  let onlyFits = false;

  const grid = el("div", { class: "grid grid-3" });
  const meta = state.benchmarkMeta || {};

  const search = el("input", {
    type: "search",
    placeholder: "Search by name, family or base model",
    style: "max-width:320px",
    oninput: (event) => { query = event.target.value.toLowerCase(); paint(); },
  });

  const tabButtons = new Map(
    ["llm", "vlm", "speech", "vision"].map((name) => [
      name,
      el("button", {
        class: name === tier ? "active" : "",
        onclick: (event) => {
          tier = name;
          for (const b of event.currentTarget.parentElement.children) b.classList.remove("active");
          event.currentTarget.classList.add("active");
          paint();
        },
      }),
    ]),
  );
  const tabs = el("div", { class: "tabs" }, [...tabButtons.values()]);

  root.append(
    el("div", { class: "row wrap", style: "margin-bottom:16px" }, [
      search,
      el("label", { class: "check" }, [
        el("input", {
          type: "checkbox",
          onchange: (event) => { onlyFits = event.target.checked; paint(); },
        }),
        "Only models that fit this machine",
      ]),
      el("span", { class: "grow" }),
      el("span", { class: "muted small" },
        meta.release ? `Certified ${meta.release} · measured ${meta.run_date}` : ""),
    ]),
    tabs,
    grid,
  );

  function card(model) {
    const bench = model.benchmark;
    const tps = bench?.tps?.["2048"] ?? bench?.tps?.["128"] ?? bench?.tps?.["0"];
    const ttft = bench?.ttft_s?.["2048"] ?? bench?.ttft_s?.["128"] ?? bench?.ttft_s?.["0"];

    const actions = el("div", { class: "actions" }, [
      el("button", {
        class: "btn btn-sm",
        text: "Details",
        onclick: () => showCard(model),
      }),
      model.tier === "llm" || model.tier === "vlm"
        ? el("button", {
            class: "btn btn-sm",
            text: model.tier === "vlm" ? "Open in Vision" : "Open in Chat",
            onclick: () => {
              localStorage.setItem(
                "hipep-chat-settings",
                JSON.stringify({
                  ...JSON.parse(localStorage.getItem("hipep-chat-settings") || "{}"),
                  model: model.id,
                }),
              );
              location.hash = model.tier === "vlm" ? "#/vision" : "#/chat";
            },
          })
        : null,
      model.installed
        ? el("button", {
            class: "btn btn-sm btn-danger",
            text: "Remove",
            onclick: async () => {
              try {
                await api.deleteLocal(model.id);
                toast(`Removed ${model.id}`, "ok");
                await refreshCatalog();
                paint();
              } catch (error) {
                toast(error.message, "error");
              }
            },
          })
        : el("button", {
            class: "btn btn-sm btn-primary",
            text: "Pull",
            onclick: async (event) => {
              event.currentTarget.disabled = true;
              try {
                const { job } = await api.pull(model.id);
                showJob(job.id, `Pulling ${model.name}`);
              } catch (error) {
                toast(error.message, "error");
                event.currentTarget.disabled = false;
              }
            },
          }),
    ]);

    const fitTag =
      model.fits === false
        ? el("span", { class: "tag tag-bad", text: "too large" })
        : model.fits === true
          ? el("span", { class: "tag tag-ok", text: "fits" })
          : null;

    return el("div", { class: "model-card" }, [
      el("div", { class: "title" }, [
        el("div", {}, [
          el("h4", { text: model.name }),
          el("div", { class: "id", text: model.id }),
        ]),
        el("span", { class: `tag tag-${model.tier}`, text: TIER_LABEL[model.tier] }),
      ]),
      el("div", { class: "specs" }, [
        el("span", { class: "tag", text: model.quant_label }),
        model.parameters ? el("span", { class: "tag", text: model.parameters }) : null,
        model.context?.max ? el("span", { class: "tag", text: `${fmtContext(model.context.max)} ctx` }) : null,
        model.footprint?.min_vram_gb
          ? el("span", { class: "tag", text: `${model.footprint.min_vram_gb} GB` })
          : null,
        fitTag,
        model.status === "failing" ? el("span", { class: "tag tag-bad", text: "failing" }) : null,
        model.status === "imported" ? el("span", { class: "tag tag-warn", text: "your import" }) : null,
        model.installed ? el("span", { class: "tag tag-ok", text: "installed" }) : null,
      ]),
      el("div", { class: "desc", text: model.description || "" }),
      tps || ttft
        ? el("div", { class: "perf" }, [
            el("div", {}, [
              el("div", { class: "k", text: "TTFT" }),
              el("div", { class: "v", text: ttft !== undefined && ttft !== null ? `${fmtNumber(ttft, 2)}s` : "-" }),
            ]),
            el("div", {}, [
              el("div", { class: "k", text: model.tier === "vision" ? "QPS" : "Tokens/s" }),
              el("div", { class: "v", text: tps ? fmtNumber(tps) : fmtNumber(bench?.qps) }),
            ]),
          ])
        : null,
      actions,
    ]);
  }

  function paint() {
    for (const [name, button] of tabButtons) {
      const count = state.catalog.filter((m) => m.tier === name).length;
      button.textContent = `${TIER_LABEL[name]} (${count})`;
    }

    let models = state.catalog.filter((m) => m.tier === tier);
    if (query) {
      models = models.filter((m) =>
        [m.id, m.name, m.family, m.base_model].some((v) => (v || "").toLowerCase().includes(query)),
      );
    }
    if (onlyFits) models = models.filter((m) => m.fits !== false);

    clear(grid);
    if (!models.length) {
      grid.append(empty("No matching models", "Adjust the search or the filter."));
      return;
    }
    models.forEach((m) => grid.append(card(m)));
  }

  function showJob(jobId, title) {
    const panel = jobPanel(jobId, {
      onFinish: async () => {
        await refreshCatalog();
        paint();
      },
    });
    modal(title, panel);
  }

  paint();

  // A port or a pull finishing elsewhere changes what belongs in the zoo, so repaint on
  // a catalog reload instead of waiting for the user to navigate away and back.
  let seen = state.catalogVersion;
  return onState(() => {
    if (state.catalogVersion === seen) return;
    seen = state.catalogVersion;
    paint();
  });
}

/* ---------- model card ---------- */

export function showCard(model) {
  const bench = model.benchmark;
  const quant = model.quantization || {};
  const rows = (pairs) =>
    el("table", { class: "data" },
      el("tbody", {}, pairs.filter(Boolean).map(([k, v]) =>
        el("tr", {}, [
          el("td", { class: "muted", style: "width:190px", text: k }),
          el("td", {}, v instanceof Node ? v : String(v ?? "-")),
        ]),
      )),
    );

  const contexts = bench ? Object.keys(bench.ttft_s || {}).filter((k) => k !== "0") : [];
  const perfTable = contexts.length
    ? el("table", { class: "data" }, [
        el("thead", {}, el("tr", {}, [
          el("th", { text: "Prompt tokens" }),
          el("th", { class: "right", text: "TTFT" }),
          el("th", { class: "right", text: "Tokens/s" }),
          el("th", { class: "right", text: "vs previous" }),
        ])),
        el("tbody", {}, contexts.map((ctx) => {
          const prevTtft = bench.previous?.ttft_s?.[ctx];
          const prevTps = bench.previous?.tps?.[ctx];
          const ttft = bench.ttft_s[ctx];
          const tps = bench.tps[ctx];
          const parts = [];
          if (prevTtft && ttft) {
            const pct = ((ttft - prevTtft) / prevTtft) * 100;
            parts.push(el("span", {
              class: pct <= 0 ? "delta-up" : "delta-down",
              text: `TTFT ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`,
            }));
          }
          if (prevTps && tps) {
            const pct = ((tps - prevTps) / prevTps) * 100;
            parts.push(el("span", {
              class: pct >= 0 ? "delta-up" : "delta-down",
              text: ` TPS ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`,
            }));
          }
          return el("tr", {}, [
            el("td", { class: "mono", text: Number(ctx).toLocaleString() }),
            el("td", { class: "num", text: ttft !== null ? `${fmtNumber(ttft, 2)} s` : "-" }),
            el("td", { class: "num", text: tps !== null ? fmtNumber(tps) : "-" }),
            el("td", { class: "num" }, parts.length ? parts : "-"),
          ]);
        })),
      ])
    : null;

  const runCommands = `# oga-server (this machine)
curl http://127.0.0.1:11434/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"model":"${model.id}","messages":[{"role":"user","content":"Hello"}]}'

# Ollama client, unchanged
ollama run ${model.id}

# hipep CLI
hipep pull ${model.id}
hipep bench ${model.id} --contexts 128,2048`;

  modal(model.name, [
    el("div", { class: "row wrap", style: "margin-bottom:16px" }, [
      el("span", { class: `tag tag-${model.tier}`, text: (TIER_LABEL[model.tier] || model.tier).toUpperCase() }),
      el("span", { class: "tag", text: model.quant_label }),
      model.capabilities?.vision ? el("span", { class: "tag tag-vlm", text: "image-text-to-text" }) : null,
      model.capabilities?.tools ? el("span", { class: "tag", text: "tool calling" }) : null,
      model.capabilities?.thinking ? el("span", { class: "tag", text: "thinking" }) : null,
      model.status === "failing"
        ? el("span", { class: "tag tag-bad", text: "failing in this release" })
        : el("span", { class: "tag tag-ok", text: model.status }),
    ]),
    el("p", { class: "muted", text: model.description || "" }),

    el("h4", { text: "Model" }),
    rows([
      ["Model id", el("code", { text: model.id })],
      ["Base model", model.base_model],
      ["Family", model.family],
      ["Parameters", model.active_parameters
        ? `${model.parameters || "?"} total, ${model.active_parameters} active`
        : model.parameters],
      ["License", model.license],
      ["Type", model.type],
    ]),

    el("h4", { text: "Quantization" }),
    rows([
      ["Scheme", (quant.scheme || "none").toUpperCase()],
      ["Weight dtype", quant.weight_dtype],
      ["Group / block size", quant.group_size ?? quant.block_size],
      ["Symmetry", quant.symmetry],
      ["Activations", quant.activation_dtype],
      quant.vision_encoder_dtype ? ["Vision encoder", quant.vision_encoder_dtype] : null,
      ["Logit pruning", quant.prune_logit ? "enabled" : "disabled"],
      ["Chunk size", quant.chunk_size],
      ["Recipe", model.recipe ? el("code", { text: model.recipe }) : "-"],
    ]),

    el("h4", { text: "Footprint and context" }),
    rows([
      ["Weights on disk", model.footprint?.weights_gb ? `${model.footprint.weights_gb} GB` : "-"],
      ["Minimum memory", model.footprint?.min_vram_gb ? `${model.footprint.min_vram_gb} GB` : "-"],
      ["Max context", model.context?.max ? model.context.max.toLocaleString() : "-"],
      ["Default context", model.context?.default ? model.context.default.toLocaleString() : "-"],
      ["Fits this machine", model.fits === null ? "unknown" : model.fits ? "yes" : "no"],
      model.local ? ["Installed at", el("code", { text: model.local.path })] : null,
      model.local ? ["Size on disk", fmtBytes(model.local.size_bytes)] : null,
    ]),

    perfTable
      ? el("div", {}, [
          el("h4", { text: `Certified performance — ${bench.release} on ${bench.device?.name || "certification host"}` }),
          perfTable,
        ])
      : null,

    el("h4", { text: "Run it" }),
    el("div", { class: "code-block" }, [
      el("pre", {}, el("code", { text: runCommands })),
    ]),
  ], { subtitle: model.base_model || model.id });
}
