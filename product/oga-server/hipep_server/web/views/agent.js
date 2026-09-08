// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Agent: tool calling, end to end and visible.
//
// The server has supported OpenAI tool calling for a while, and nothing in the console
// exercised it, so the only way to see whether a model on this machine can actually
// drive a tool was to write a client. This view is that client: it sends the tool
// schemas, runs the tools the model asks for, feeds the results back, and repeats until
// the model answers without asking for another one.
//
// The tools are real (see tools.js) and they run in the browser, so a round trip that
// works here is a round trip, not a rehearsal. The whole exchange is shown rather than
// summarised, because the interesting failures are a model that invents a tool, one
// that emits arguments that are nearly JSON, and one that never stops calling.

import { api } from "../api.js";
import { state } from "../app.js";
import { TOOLS, runToolCall } from "../tools.js";
import { clear, el, empty, fill, fmtDuration, fmtNumber, renderMarkdown, toast } from "../ui.js";

const SETTINGS_KEY = "hipep-agent-settings";

// A model that keeps asking for tools would otherwise loop until the tab is closed.
// Eight is well past any of the shipped tools needing each other and short enough that
// a runaway is a wait rather than a hang.
const MAX_ROUNDS = 8;

const defaults = {
  model: null,
  enabled: TOOLS.map((t) => t.name),
  system:
    "You are a helpful assistant with access to tools. When a tool can answer part " +
    "of the question, call it rather than guessing. After you have the results, " +
    "answer the user directly.",
  maxTokens: 512,
};

function loadSettings() {
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") };
  } catch {
    return { ...defaults };
  }
}

function saveSettings(settings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}

export async function renderAgent(root) {
  const settings = loadSettings();
  const candidates = state.catalog.filter((m) => m.tier === "llm" || m.tier === "vlm");

  if (!candidates.length) {
    root.append(empty("No models available", "The catalog is empty or the server is unreachable."));
    return;
  }

  let current =
    candidates.find((m) => m.id === settings.model) ||
    candidates.find((m) => m.installed) ||
    candidates[0];

  // The transcript the model sees. Rendering reads the same list, so what is on screen
  // is what was sent rather than a parallel account of it.
  let transcript = [];
  let running = false;
  let rounds = 0;
  let lastUsage = null;

  // --- layout ---

  const log = el("div", { class: "agent-log" });
  const textarea = el("textarea", {
    rows: 1,
    placeholder: "Ask something a tool can answer.  Enter to send, Shift+Enter for a new line.",
  });
  const runBtn = el("button", { class: "btn btn-primary", text: "Run" });
  const roundHint = el("span", { class: "composer-hint" });

  const composer = el("div", { class: "composer" }, [
    el("div", { class: "composer-box" }, [
      textarea,
      el("div", { class: "composer-actions" }, [
        el("button", {
          class: "btn btn-ghost btn-sm",
          text: "Clear",
          onclick: () => {
            transcript = [];
            rounds = 0;
            lastUsage = null;
            paintLog();
            paintMetrics();
          },
        }),
        roundHint,
        runBtn,
      ]),
    ]),
  ]);

  const modelSelect = el(
    "select",
    {
      onchange: (event) => {
        current = candidates.find((m) => m.id === event.target.value);
        settings.model = current.id;
        saveSettings(settings);
        paintModelNote();
      },
    },
    candidates.map((m) =>
      el(
        "option",
        { value: m.id, selected: m.id === current.id },
        `${m.name}${m.installed ? "" : "  (not downloaded)"}`,
      ),
    ),
  );

  const modelNote = el("div", { class: "col small" });
  const toolList = el("div", { class: "tool-list" });
  const metrics = el("div", { class: "live-metrics" });

  const systemInput = el("textarea", {
    rows: 5,
    oninput: (event) => {
      settings.system = event.target.value;
      saveSettings(settings);
    },
  });
  systemInput.value = settings.system;

  const side = el("aside", { class: "chat-side" }, [
    el("div", {}, [el("h4", { text: "Model" }), modelSelect, modelNote]),
    el("div", {}, [
      el("h4", { text: "Tools" }),
      el("div", { class: "faint", style: "margin-bottom:8px" },
        "These run in your browser. Uncheck one to hide it from the model."),
      toolList,
    ]),
    el("div", {}, [el("h4", { text: "Last run" }), metrics]),
    el("div", {}, [el("h4", { text: "System prompt" }), systemInput]),
  ]);

  root.append(
    el("div", { class: "chat" }, [
      el("div", { class: "chat-main" }, [log, composer]),
      side,
    ]),
  );

  // --- painting ---

  function paintTools() {
    fill(
      toolList,
      ...TOOLS.map((tool) => {
        const box = el("input", {
          type: "checkbox",
          checked: settings.enabled.includes(tool.name),
          onchange: (event) => {
            settings.enabled = event.target.checked
              ? [...settings.enabled, tool.name]
              : settings.enabled.filter((n) => n !== tool.name);
            saveSettings(settings);
          },
        });
        return el("label", { class: "tool-row" }, [
          box,
          el("div", {}, [
            el("code", { text: tool.name }),
            el("div", { class: "faint", text: tool.schema.function.description }),
          ]),
        ]);
      }),
    );
  }

  function paintModelNote() {
    // The catalog says whether a model was certified for tool use. A model without the
    // capability is still worth allowing -- plenty of them can be talked into it -- so
    // this is a warning and not a restriction.
    fill(
      modelNote,
      el("div", { class: "row wrap", style: "margin-top:10px" }, [
        el("span", { class: `tag tag-${current.tier}`, text: current.tier.toUpperCase() }),
        el("span", { class: "tag", text: current.quant_label }),
        current.capabilities?.tools ? el("span", { class: "tag", text: "tools" }) : null,
      ]),
      current.capabilities?.tools
        ? null
        : el("div", { class: "notice", style: "margin-top:10px" },
            "This model is not certified for tool use. It may ignore the tools or " +
            "describe calling one instead of calling it."),
      current.installed
        ? null
        : el("div", { class: "notice", style: "margin-top:10px" },
            "Not downloaded. Pull it from the Models tab first."),
    );
  }

  function paintMetrics() {
    const cells = [
      ["Rounds", rounds || "-"],
      ["TTFT", lastUsage?.ttft_s !== undefined ? fmtDuration(lastUsage.ttft_s) : "-"],
      ["Tokens/s", lastUsage?.tps ? fmtNumber(lastUsage.tps) : "-"],
      ["Generated", lastUsage?.completion_tokens ?? "-"],
    ];
    fill(
      metrics,
      ...cells.map(([k, v]) =>
        el("div", { class: "m" }, [
          el("div", { class: "k", text: k }),
          el("div", { class: "v", text: String(v) }),
        ]),
      ),
    );
  }

  function callNode(call, result) {
    const args = call.function?.arguments || "{}";
    return el("div", { class: "tool-call" }, [
      el("div", { class: "tool-call-head" }, [
        el("span", { class: "tag", text: "tool call" }),
        el("code", { text: call.function?.name || "?" }),
      ]),
      el("pre", { class: "tool-args", text: args }),
      result === undefined
        ? el("div", { class: "faint", text: "running..." })
        : el("pre", { class: "tool-result", text: result }),
    ]);
  }

  function messageNode(entry) {
    if (entry.kind === "tool-round") {
      return el("div", { class: "msg assistant" }, [
        el("div", { class: "who", text: "EP" }),
        el("div", { class: "body" }, [
          entry.thinking
            ? (() => {
                const md = el("div", { class: "md" });
                renderMarkdown(md, entry.thinking);
                return md;
              })()
            : null,
          ...entry.calls.map((call, i) => callNode(call, entry.results[i])),
        ]),
      ]);
    }

    const body = el("div", { class: "body" });
    const md = el("div", { class: "md" });
    renderMarkdown(md, entry.content || "");
    body.append(md);
    if (entry.usage) {
      body.append(
        el("div", { class: "meta" }, [
          el("span", { text: `TTFT ${fmtDuration(entry.usage.ttft_s)}` }),
          el("span", { text: `${fmtNumber(entry.usage.tps)} tok/s` }),
          el("span", { text: `${entry.usage.completion_tokens} tokens` }),
        ]),
      );
    }
    return el("div", { class: `msg ${entry.role}` }, [
      el("div", { class: "who", text: entry.role === "user" ? "YOU" : "EP" }),
      body,
    ]);
  }

  // The screen is rebuilt from `rendered`, a view of the transcript that groups each
  // assistant tool call with the result that answered it. Reading the raw transcript
  // instead would show a tool result as a message from nobody.
  let rendered = [];

  function paintLog() {
    clear(log);
    if (!rendered.length) {
      log.append(
        empty(
          "Give the model a job",
          "Ask something the tools on the right can answer, such as " +
            '"what time is it, and what GPU is this?". Every tool call and its result ' +
            "is shown, so you can see what the model asked for and what it got back.",
        ),
      );
      return;
    }
    rendered.forEach((entry) => log.append(messageNode(entry)));
    log.scrollTop = log.scrollHeight;
  }

  function setRunning(value) {
    running = value;
    runBtn.disabled = value;
    runBtn.textContent = value ? "Working" : "Run";
    textarea.disabled = value;
  }

  // --- the loop ---

  async function run() {
    const prompt = textarea.value.trim();
    if (!prompt || running) return;
    textarea.value = "";
    textarea.style.height = "auto";

    if (settings.system && !transcript.length) {
      transcript.push({ role: "system", content: settings.system });
    }
    transcript.push({ role: "user", content: prompt });
    rendered.push({ kind: "message", role: "user", content: prompt });
    paintLog();

    const tools = TOOLS.filter((t) => settings.enabled.includes(t.name)).map((t) => t.schema);
    setRunning(true);
    rounds = 0;

    try {
      for (let round = 0; round < MAX_ROUNDS; round += 1) {
        roundHint.textContent = `round ${round + 1}`;
        const response = await api.chat({
          model: current.id,
          messages: transcript,
          max_tokens: settings.maxTokens,
          ...(tools.length ? { tools } : {}),
        });

        const message = response.choices?.[0]?.message || {};
        lastUsage = response.usage || lastUsage;
        rounds = round + 1;
        paintMetrics();

        const calls = message.tool_calls || [];
        if (!calls.length) {
          transcript.push({ role: "assistant", content: message.content || "" });
          rendered.push({
            kind: "message",
            role: "assistant",
            content: message.content || "",
            usage: response.usage,
          });
          paintLog();
          return;
        }

        // The assistant turn that asked for the tools has to go into the transcript
        // exactly as received: a tool result without the call it answers is a protocol
        // error, and models that notice it tend to repeat the call forever.
        transcript.push({
          role: "assistant",
          content: message.content ?? null,
          tool_calls: calls,
        });

        const group = {
          kind: "tool-round",
          thinking: message.content || "",
          calls,
          results: new Array(calls.length).fill(undefined),
        };
        rendered.push(group);
        paintLog();

        for (const [index, call] of calls.entries()) {
          const result = await runToolCall(call);
          group.results[index] = result;
          transcript.push({
            role: "tool",
            tool_call_id: call.id,
            name: call.function?.name,
            content: result,
          });
          paintLog();
        }
      }

      // Falling out of the loop means the model asked for a tool every time. Saying so
      // is more useful than an empty answer, and the transcript above shows what it
      // kept asking for.
      rendered.push({
        kind: "message",
        role: "assistant",
        content:
          `The model asked for a tool on all ${MAX_ROUNDS} rounds without answering. ` +
          "Small models sometimes call the same tool repeatedly; try a larger one, or " +
          "fewer tools.",
      });
      paintLog();
    } catch (error) {
      toast(error.message || "the agent run failed", "bad");
      rendered.push({
        kind: "message",
        role: "assistant",
        content: `**The run failed.** ${error.message || error}`,
      });
      paintLog();
    } finally {
      roundHint.textContent = "";
      setRunning(false);
    }
  }

  runBtn.onclick = run;
  textarea.oninput = () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  };
  textarea.onkeydown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      run();
    }
  };

  paintTools();
  paintModelNote();
  paintMetrics();
  paintLog();
}
