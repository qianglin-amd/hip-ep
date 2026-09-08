// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Application shell: hash router, shared state, live status header.

import { api, subscribe } from "./api.js";
import { $, $$, clear, el, fill, toast } from "./ui.js";

import { renderAgent } from "./views/agent.js";
import { renderChat } from "./views/chat.js";
import { renderModels } from "./views/models.js";
import { renderPort } from "./views/port.js";
import { renderBenchmark } from "./views/benchmark.js";
import { renderServer } from "./views/server.js";
import { renderDevelop } from "./views/develop.js";

export const state = {
  status: null,
  catalog: [],
  // Bumped on every catalog reload so a view can tell a catalog change from the status
  // ticks that also notify listeners, and repaint only when its data actually changed.
  catalogVersion: 0,
  benchmarkMeta: {},
  tiers: [],
  jobs: new Map(),
  listeners: new Set(),
};

export function onState(fn) {
  state.listeners.add(fn);
  return () => state.listeners.delete(fn);
}

function emit() {
  for (const fn of state.listeners) {
    try { fn(state); } catch (error) { console.error(error); }
  }
}

const VIEWS = {
  chat: { title: "Chat", subtitle: "Talk to a certified model", render: renderChat },
  vision: { title: "Vision", subtitle: "Image and text with a VLM", render: (r) => renderChat(r, { vision: true }) },
  agent: { title: "Agent", subtitle: "Tool calling, with the tools running here", render: renderAgent },
  models: { title: "Models", subtitle: "Certified model zoo", render: renderModels },
  port: { title: "Port a model", subtitle: "Bring your own weights", render: renderPort },
  benchmark: { title: "Benchmarks", subtitle: "Published results and local runs", render: renderBenchmark },
  server: { title: "Server", subtitle: "Live load, sessions and hardware", render: renderServer },
  develop: { title: "Develop", subtitle: "Client samples in C++, Python and WinML", render: renderDevelop },
};

let disposeView = null;

async function route() {
  const name = (location.hash.replace(/^#\/?/, "") || "chat").split("/")[0];
  const view = VIEWS[name] || VIEWS.chat;

  for (const link of $$("#nav a")) {
    link.classList.toggle("active", link.dataset.view === name);
  }
  fill($("#crumb"), view.title, el("span", { text: view.subtitle }));

  const root = $("#view");
  root.classList.toggle("flush", name === "chat" || name === "vision");
  clear(root);

  if (typeof disposeView === "function") {
    try { disposeView(); } catch (error) { console.error(error); }
  }
  disposeView = null;

  try {
    disposeView = await view.render(root);
  } catch (error) {
    console.error(error);
    fill(root, 
      el("div", { class: "notice bad" }, [
        el("strong", { text: "This view failed to load. " }),
        error.message,
      ]),
    );
  }
}

/* ---------- header ---------- */

function paintStatus(status) {
  const badge = $("#engine-badge");
  const engine = status?.engine || {};
  const live = engine.real === true;
  badge.className = `badge ${live ? "badge-live" : "badge-replay"}`;
  badge.title = status?.engine_reason || "";
  fill(badge, 
    el("span", { class: "dot" }),
    el("span", { class: "label", text: live ? "GPU engine" : "Replay engine" }),
  );

  const gpu = status?.host?.gpus?.[0];
  $("#device-line").textContent = gpu
    ? `${gpu.profile_name} · ${gpu.gfx_arch}${gpu.memory_gb ? ` · ${gpu.memory_gb} GB` : ""}`
    : "No AMD GPU detected";

  const load = status?.load || {};
  const pill = $("#load-pill");
  const parts = [`${load.active_requests ?? 0} active`];
  if (load.queued_requests) parts.push(`${load.queued_requests} queued`);
  parts.push(`${load.sessions ?? 0} session${load.sessions === 1 ? "" : "s"}`);
  if (load.loaded_models) parts.push(`${load.loaded_models} loaded`);
  pill.textContent = parts.join("  ·  ");
}

function paintActivity() {
  const running = [...state.jobs.values()].filter((j) => j.state === "running");
  const node = $("#activity");
  node.hidden = running.length === 0;
  if (!running.length) return;
  const job = running[0];
  fill(node, 
    el("span", { text: job.title }),
    el("div", { class: "bar" }, el("i", { style: `width:${(job.progress * 100).toFixed(0)}%` })),
    el("span", {
      class: "faint",
      text: running.length > 1 ? `+${running.length - 1}` : `${(job.progress * 100).toFixed(0)}%`,
    }),
  );
}

/* ---------- boot ---------- */

async function refreshCatalog() {
  const data = await api.catalog();
  state.catalog = data.models;
  state.benchmarkMeta = data.benchmark;
  state.tiers = data.tiers;
  state.catalogVersion += 1;
  emit();
}

let statusStream = null;
let reconnectTimer = null;

function connectStatus() {
  statusStream?.close();
  statusStream = subscribe("/v1/status/events?interval=2", {
    onMessage: (status) => {
      state.status = status;
      paintStatus(status);
      emit();
    },
    onError: () => {
      const badge = $("#engine-badge");
      badge.className = "badge badge-error";
      fill(badge, el("span", { class: "dot" }), el("span", { text: "disconnected" }));
      // The server may simply be restarting. Retry rather than making the user
      // reload, but keep a single timer so failures cannot stack up reconnects.
      if (reconnectTimer === null) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connectStatus();
        }, 3000);
      }
    },
  });
}

function connectJobs() {
  return subscribe("/v1/jobs/events", {
    onMessage: (job) => {
      state.jobs.set(job.id, job);
      paintActivity();
      emit();
    },
  });
}

function initTheme() {
  const saved = localStorage.getItem("hipep-theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("hipep-theme", next);
  });
}

async function main() {
  initTheme();
  window.addEventListener("hashchange", route);

  try {
    state.status = await api.status();
    paintStatus(state.status);
    await refreshCatalog();
  } catch (error) {
    toast(`Cannot reach oga-server: ${error.message}`, "error", 9000);
  }

  connectStatus();
  connectJobs();
  await route();
}

export { refreshCatalog };
main();
