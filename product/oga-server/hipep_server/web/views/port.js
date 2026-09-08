// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Bring your own model: point at a Hugging Face repo, pick a recipe, watch the
// fetch/export/quantize/convert/validate/benchmark/publish pipeline run.

import { api } from "../api.js";
import { refreshCatalog, state } from "../app.js";
import { clear, el, fill, toast } from "../ui.js";
import { jobPanel, jobRow } from "./jobview.js";

const EXAMPLES = [
  "meta-llama/Llama-3.2-3B-Instruct",
  "Qwen/Qwen2.5-7B-Instruct",
  "google/gemma-3-4b-it",
  "microsoft/Phi-4-mini-instruct",
];

export async function renderPort(root) {
  let recipes = [];
  try {
    recipes = (await api.recipes()).recipes;
  } catch (error) {
    toast(`Could not load recipes: ${error.message}`, "error");
  }

  const panelHost = el("div", { class: "col", style: "gap:16px" });
  const historyHost = el("div", {});

  const sourceInput = el("input", {
    placeholder: "org/model on Hugging Face, a URL, or a local path",
    autofocus: true,
  });
  const nameInput = el("input", { placeholder: "Optional display name" });

  const DEFAULT_RECIPE = "int4-rtn-g128";
  const recipeSelect = el("select", {},
    recipes.length
      ? recipes.map((r) =>
          // A recipe needing a calibration pass this server cannot run stays listed and
          // readable -- it documents how a certified artifact was made -- but it cannot
          // be chosen, because choosing it could only ever fail.
          el("option", {
            value: r.id,
            title: r.unavailable_reason || r.description || "",
            selected: r.id === DEFAULT_RECIPE,
            disabled: r.buildable === false,
          }, `${r.name || r.id}${r.buildable === false ? " (needs calibration — not portable here)" : r.recommended_for ? ` — ${r.recommended_for}` : ""}`),
        )
      : [el("option", { value: DEFAULT_RECIPE }, "INT4 RTN, group size 128 (built-in default)")],
  );

  const chunkInput = el("input", { type: "number", min: "1", value: "512" });
  const pruneInput = el("input", { type: "checkbox", checked: true });
  const calibInput = el("input", { type: "number", min: "1", value: "128" });
  const benchInput = el("input", { type: "checkbox", checked: true });

  const recipeNote = el("div", { class: "faint small" });
  function paintRecipeNote() {
    const recipe = recipes.find((r) => r.id === recipeSelect.value);
    recipeNote.textContent = recipe?.unavailable_reason || recipe?.description || "";
  }
  recipeSelect.addEventListener("change", paintRecipeNote);
  paintRecipeNote();

  // A port downloads tens of gigabytes before it can tell you anything. Preflight reads
  // the model's config alone -- a few kilobytes -- so the answer arrives while the user
  // is still filling the form.
  const preflight = el("div", { class: "faint small" });
  let preflightFor = "";
  async function runPreflight() {
    const source = sourceInput.value.trim();
    if (!source || source === preflightFor) return;
    preflightFor = source;
    preflight.textContent = "Checking the model…";
    preflight.className = "faint small";
    try {
      const v = await api.portPreflight(source);
      if (sourceInput.value.trim() !== source) return;
      const lines = [
        `${v.architecture}${v.hidden_size ? `, hidden size ${v.hidden_size}` : ""}` +
          `${v.context ? `, ${v.context.toLocaleString()} token context` : ""}`,
        `Suggested recipe: ${v.recipe} at block size ${v.block_size}.`,
        ...(v.verified
          ? [`Ported and checked here on ${v.verified.gpu} using ${v.verified.checkpoint}.`]
          : []),
        ...v.warnings,
        ...v.notes,
      ];
      // Warnings and notes are written as fragments so they can be read on their own;
      // strung together into a paragraph they have to start like sentences.
      preflight.textContent = lines
        .map((line) => line.charAt(0).toUpperCase() + line.slice(1))
        .join(" ");
      preflight.className = v.warnings.length ? "notice" : "faint small";
    } catch (error) {
      if (sourceInput.value.trim() !== source) return;
      preflight.textContent = error.message;
      preflight.className = "notice bad";
      preflightFor = "";
    }
  }
  sourceInput.addEventListener("change", runPreflight);
  sourceInput.addEventListener("blur", runPreflight);

  const submit = el("button", { class: "btn btn-primary", text: "Start porting" });

  const form = el("div", { class: "card" }, [
    el("h3", { text: "Source" }),
    el("label", { class: "field" }, ["Model", sourceInput, preflight]),
    el("div", { class: "row wrap", style: "gap:6px" },
      EXAMPLES.map((example) =>
        el("button", {
          class: "btn btn-ghost btn-sm",
          text: example,
          onclick: () => { sourceInput.value = example; runPreflight(); },
        }),
      ),
    ),
    el("label", { class: "field" }, ["Name", nameInput]),

    el("h3", { style: "margin-top:20px", text: "Quantization" }),
    el("label", { class: "field" }, ["Recipe", recipeSelect, recipeNote]),
    el("div", { class: "grid grid-2" }, [
      el("label", { class: "field" }, [
        "Chunk size",
        chunkInput,
        el("div", { class: "faint small", text: "Quark processes the model in chunks; 512 is the project default." }),
      ]),
      el("label", { class: "field" }, [
        "Calibration samples",
        calibInput,
        el("div", { class: "faint small", text: "Used by AWQ and other data-aware schemes." }),
      ]),
    ]),
    el("label", { class: "check" }, [pruneInput, "Prune the logit head (recommended)"]),
    el("label", { class: "check" }, [benchInput, "Benchmark the result when the port finishes"]),

    el("div", { class: "row", style: "margin-top:20px" }, [
      submit,
      el("span", { class: "faint small" },
        state.status?.engine?.real
          ? "Runs on this machine's GPU."
          : "The replay engine simulates each stage with realistic timing; no artifacts are written."),
    ]),
  ]);

  root.append(
    el("div", { class: "grid grid-2 grid-wide" }, [
      form,
      el("div", { class: "col", style: "gap:16px" }, [
        panelHost,
        el("div", { class: "card" }, [
          el("h3", { text: "Recent port jobs" }),
          historyHost,
        ]),
      ]),
    ]),
  );

  let openPanel = null;

  function show(jobId, title) {
    openPanel?.close?.();
    const panel = jobPanel(jobId, {
      onFinish: async (job) => {
        if (job.state === "succeeded") {
          toast(`${title} finished`, "ok");
          await refreshCatalog();
        } else if (job.state === "failed") {
          toast(`${title} failed`, "error", 8000);
        }
        paintHistory();
      },
    });
    openPanel = panel;
    fill(panelHost, 
      el("div", { class: "card" }, [el("h3", { text: title }), panel]),
    );
  }

  submit.addEventListener("click", async () => {
    const source = sourceInput.value.trim();
    if (!source) {
      toast("Enter a model source first", "error");
      sourceInput.focus();
      return;
    }
    submit.disabled = true;
    try {
      const { job } = await api.port({
        source,
        name: nameInput.value.trim() || null,
        quantization: recipeSelect.value,
        chunk_size: Number(chunkInput.value) || 512,
        prune_logit: pruneInput.checked,
        calibration_samples: Number(calibInput.value) || 128,
        benchmark: benchInput.checked,
      });
      show(job.id, `Porting ${source}`);
      paintHistory();
    } catch (error) {
      toast(error.message, "error", 8000);
    } finally {
      submit.disabled = false;
    }
  });

  async function paintHistory() {
    try {
      const { jobs } = await api.jobs("port");
      clear(historyHost);
      if (!jobs.length) {
        historyHost.append(
          el("p", { class: "muted small" },
            "Nothing yet. A finished port is registered locally and appears in the Models tab."),
        );
        return;
      }
      historyHost.append(
        el("table", { class: "data" },
          el("tbody", {}, jobs.map((job) =>
            jobRow(job, { onOpen: () => show(job.id, job.title) }),
          )),
        ),
      );
    } catch (error) {
      fill(historyHost, el("div", { class: "notice bad", text: error.message }));
    }
  }

  await paintHistory();

  // Reattach to a port that is still running, so navigating away and back does not
  // lose the view of an in-flight job.
  const running = [...state.jobs.values()].find(
    (j) => j.kind === "port" && j.state === "running",
  );
  if (running) show(running.id, running.title);

  return () => openPanel?.close?.();
}
