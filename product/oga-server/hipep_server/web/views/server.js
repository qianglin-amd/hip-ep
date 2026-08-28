// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Server page: what is running right now, who is running it, and what this machine is.
// Repaints from the shared status stream rather than opening a second subscription.

import { api } from "../api.js";
import { onState, state } from "../app.js";
import {
  clear, el, fill, fmtBytes, fmtDuration, fmtNumber, timeAgo, toast,
} from "../ui.js";
import { jobRow } from "./jobview.js";

export async function renderServer(root) {
  const stats = el("div", { class: "grid grid-4" });
  const activeHost = el("div", {});
  const sessionHost = el("div", {});
  const recentHost = el("div", {});
  const hostHost = el("div", {});
  const jobHost = el("div", {});

  root.append(
    stats,
    el("div", { class: "grid grid-2 grid-wide", style: "margin-top:20px" }, [
      el("div", { class: "card" }, [
        el("h3", { text: "In flight" }),
        el("p", { class: "muted small" },
          "Every request currently admitted or waiting, across all clients."),
        activeHost,
      ]),
      el("div", { class: "card" }, [
        el("h3", { text: "Clients" }),
        el("p", { class: "muted small" },
          "One row per session. Clients that send an X-HiPEP-Session header keep a " +
          "stable identity; everyone else is grouped by address and user agent."),
        sessionHost,
      ]),
    ]),
    el("div", { class: "grid grid-2 grid-wide", style: "margin-top:20px" }, [
      el("div", { class: "card" }, [el("h3", { text: "This machine" }), hostHost]),
      el("div", { class: "card" }, [el("h3", { text: "Recent completions" }), recentHost]),
    ]),
    el("div", { class: "card", style: "margin-top:20px" }, [
      el("h3", { text: "Jobs" }),
      jobHost,
    ]),
  );

  function paintStats(status) {
    const load = status.load || {};
    const totals = status.totals || {};
    const limits = status.limits || {};
    const process = status.process || {};

    const card = (label, value, detail) =>
      el("div", { class: "card stat-card" }, [
        el("div", { class: "k", text: label }),
        el("div", { class: "v", text: value }),
        detail ? el("div", { class: "faint small", text: detail }) : null,
      ]);

    fill(stats, 
      card(
        "Active requests",
        String(load.active_requests ?? 0),
        `${load.queued_requests ?? 0} queued of ${limits.max_queue_depth} allowed`,
      ),
      card(
        "Loaded models",
        `${load.loaded_models ?? 0} / ${limits.max_loaded_models ?? "-"}`,
        (status.engine?.loaded || []).join(", ") || "nothing resident",
      ),
      card(
        "Tokens served",
        fmtNumber(totals.completion_tokens ?? 0, 0),
        `${fmtNumber(totals.prompt_tokens ?? 0, 0)} prompt tokens read`,
      ),
      card(
        "Requests",
        String(totals.requests ?? 0),
        `${totals.errors ?? 0} failed, ${totals.rejected ?? 0} rejected`,
      ),
      card("Uptime", fmtDuration(status.uptime_s ?? 0), `version ${status.version || "-"}`),
      card(
        "Engine",
        status.engine?.real ? "GPU" : "Replay",
        status.engine_reason || status.engine?.name || "",
      ),
      card(
        "Host memory",
        process.memory_total_gb ? `${fmtNumber(process.memory_used_gb, 1)} / ${process.memory_total_gb} GB` : "-",
        process.cpu_percent !== undefined ? `CPU ${process.cpu_percent}%` : "",
      ),
      card(
        "Model storage",
        fmtBytes(status.storage?.bytes ?? 0),
        `${status.storage?.models ?? 0} installed in ${status.storage?.root || ""}`,
      ),
    );
  }

  function paintActive(status) {
    const active = status.active || [];
    clear(activeHost);
    if (!active.length) {
      activeHost.append(el("p", { class: "faint small", text: "Idle." }));
      return;
    }
    activeHost.append(
      el("table", { class: "data" }, [
        el("thead", {}, el("tr", {}, [
          el("th", { text: "Model" }),
          el("th", { text: "State" }),
          el("th", { text: "Client" }),
          el("th", { class: "right", text: "Wait" }),
          el("th", { class: "right", text: "TTFT" }),
          el("th", { class: "right", text: "tok/s" }),
          el("th", { class: "right", text: "Tokens" }),
        ])),
        el("tbody", {}, active.map((request) =>
          el("tr", {}, [
            el("td", { class: "mono small", text: request.model }),
            el("td", {},
              el("span", { class: `badge badge-${request.state === "queued" ? "pending" : "running"}` }, [
                el("span", { class: "dot" }),
                el("span", { text: request.state }),
              ]),
            ),
            el("td", { class: "faint small", text: `${request.client} · ${request.api}` }),
            el("td", { class: "num", text: `${fmtNumber(request.wait_s, 2)}s` }),
            el("td", { class: "num", text: request.ttft_s ? `${fmtNumber(request.ttft_s, 2)}s` : "-" }),
            el("td", { class: "num", text: request.tps ? fmtNumber(request.tps) : "-" }),
            el("td", { class: "num", text: String(request.tokens) }),
          ]),
        )),
      ]),
    );
  }

  function paintSessions(status) {
    const sessions = status.sessions_detail || [];
    clear(sessionHost);
    if (!sessions.length) {
      sessionHost.append(el("p", { class: "faint small", text: "No clients have connected yet." }));
      return;
    }
    sessionHost.append(
      el("table", { class: "data" }, [
        el("thead", {}, el("tr", {}, [
          el("th", { text: "Session" }),
          el("th", { text: "Client" }),
          el("th", { class: "right", text: "Requests" }),
          el("th", { class: "right", text: "Tokens" }),
          el("th", { class: "right", text: "Idle" }),
        ])),
        el("tbody", {}, sessions.map((session) =>
          el("tr", {}, [
            el("td", {}, [
              el("div", { class: "mono small", text: session.id }),
              el("div", { class: "faint small", text: session.api }),
            ]),
            el("td", { class: "faint small" }, [
              el("div", { text: session.client }),
              el("div", { class: "truncate", title: session.user_agent, text: session.user_agent }),
            ]),
            el("td", { class: "num", text: String(session.requests) }),
            el("td", { class: "num", text: fmtNumber(session.completion_tokens, 0) }),
            el("td", { class: "num faint", text: fmtDuration(session.idle_s) }),
          ]),
        )),
      ]),
    );
  }

  function paintRecent(status) {
    const recent = status.recent || [];
    clear(recentHost);
    if (!recent.length) {
      recentHost.append(el("p", { class: "faint small", text: "No completed requests yet." }));
      return;
    }
    recentHost.append(
      el("table", { class: "data" }, [
        el("thead", {}, el("tr", {}, [
          el("th", { text: "Model" }),
          el("th", { class: "right", text: "TTFT" }),
          el("th", { class: "right", text: "tok/s" }),
          el("th", { class: "right", text: "Tokens" }),
          el("th", { class: "right", text: "When" }),
        ])),
        el("tbody", {}, recent.slice(0, 12).map((row) =>
          el("tr", {}, [
            el("td", { class: "mono small", text: row.model }),
            el("td", { class: "num", text: row.ttft_s ? `${fmtNumber(row.ttft_s, 2)}s` : "-" }),
            el("td", { class: "num", text: row.tps ? fmtNumber(row.tps) : "-" }),
            el("td", { class: "num", text: String(row.completion_tokens ?? "-") }),
            el("td", { class: "num faint", text: timeAgo(row.at) }),
          ]),
        )),
      ]),
    );
  }

  function paintHost(status) {
    const host = status.host || {};
    const gpus = host.gpus || [];
    fill(hostHost, 
      el("div", { class: "kv" }, [
        el("span", { class: "k", text: "host" }),
        el("span", { class: "v", text: host.hostname || "-" }),
        el("span", { class: "k", text: "operating system" }),
        el("span", { class: "v", text: `${host.os || "-"} ${host.os_version || ""}`.trim() }),
        el("span", { class: "k", text: "cpu" }),
        el("span", { class: "v" },
          `${host.cpu || "unknown"}${host.cpu_cores ? ` · ${host.cpu_cores} cores` : ""}` +
          `${host.memory_gb ? ` · ${host.memory_gb} GB` : ""}`),
        el("span", { class: "k", text: "architecture" }),
        el("span", { class: "v", text: host.arch || "-" }),
        el("span", { class: "k", text: "python" }),
        el("span", { class: "v", text: host.python || "-" }),
        el("span", { class: "k", text: "certified release" }),
        el("span", { class: "v", text: `${status.catalog?.release || "-"} (${status.catalog?.run_date || "-"})` }),
      ]),
      gpus.length
        ? el("div", { class: "col", style: "gap:8px;margin-top:14px" },
            gpus.map((gpu) =>
              el("div", { class: "gpu-row" }, [
                el("div", {}, [
                  el("strong", { text: gpu.profile_name || gpu.name }),
                  el("div", { class: "faint small", text: gpu.name }),
                ]),
                el("span", { class: "grow" }),
                el("span", { class: "tag", text: gpu.gfx_arch || "unknown arch" }),
                gpu.memory_gb ? el("span", { class: "tag", text: `${gpu.memory_gb} GB` }) : null,
              ]),
            ),
          )
        : el("div", { class: "notice", style: "margin-top:14px" }, [
            el("strong", { text: "No AMD GPU detected. " }),
            "The server is serving replayed responses so the product remains usable, " +
            "but nothing is executing on hardware.",
          ]),
      el("div", { class: "row", style: "margin-top:14px" }, [
        el("button", {
          class: "btn btn-sm",
          text: "Re-detect hardware",
          onclick: async (event) => {
            event.currentTarget.disabled = true;
            try {
              const fresh = await api.hardware();
              toast(
                fresh.gpus?.length
                  ? `Found ${fresh.gpus.length} GPU: ${fresh.gpus[0].profile_name}`
                  : "No AMD GPU found",
                fresh.gpus?.length ? "ok" : "info",
              );
            } catch (error) {
              toast(error.message, "error");
            } finally {
              event.currentTarget.disabled = false;
            }
          },
        }),
      ]),
    );
  }

  function paintJobs() {
    const jobs = [...state.jobs.values()].sort((a, b) => b.created_at - a.created_at);
    clear(jobHost);
    if (!jobs.length) {
      jobHost.append(el("p", { class: "faint small", text: "No jobs have run in this session." }));
      return;
    }
    jobHost.append(
      el("table", { class: "data" }, el("tbody", {}, jobs.slice(0, 12).map((job) => jobRow(job)))),
    );
  }

  function paint() {
    const status = state.status;
    if (!status) return;
    paintStats(status);
    paintActive(status);
    paintSessions(status);
    paintRecent(status);
    paintHost(status);
    paintJobs();
  }

  paint();
  return onState(paint);
}
