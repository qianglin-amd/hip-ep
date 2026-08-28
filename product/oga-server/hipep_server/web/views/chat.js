// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Chat and Vision. One view: the Vision entry point simply preselects a VLM and shows
// the image affordances, because the underlying request is the same multimodal chat
// call and maintaining two near-identical views would guarantee they drift.

import { fileToDataURL, streamChat } from "../api.js";
import { state } from "../app.js";
import {
  $, clear, copyButton, el, empty, fill, fmtDuration, fmtNumber, renderMarkdown, toast,
} from "../ui.js";

const SETTINGS_KEY = "hipep-chat-settings";

const defaults = {
  model: null,
  system: "",
  temperature: 0.7,
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

export async function renderChat(root, { vision = false } = {}) {
  const settings = loadSettings();
  const models = state.catalog.filter((m) => m.tier === "llm" || m.tier === "vlm");
  const candidates = vision ? models.filter((m) => m.capabilities?.vision) : models;

  if (!candidates.length) {
    root.append(empty("No models available", "The catalog is empty or the server is unreachable."));
    return;
  }

  let current =
    candidates.find((m) => m.id === settings.model) ||
    candidates.find((m) => m.installed) ||
    candidates[0];

  /** @type {{role:string,content:string,images:string[],usage?:object}[]} */
  const history = [];
  let attachments = [];
  let active = null;

  // --- layout ---

  const log = el("div", { class: "chat-log" });
  const textarea = el("textarea", {
    rows: 1,
    placeholder: vision
      ? "Ask about an image — drop, paste or browse to attach one"
      : "Send a message.  Enter to send, Shift+Enter for a new line.",
  });
  const attachmentBar = el("div", { class: "attachments", hidden: true });
  const fileInput = el("input", { type: "file", accept: "image/*", multiple: true, hidden: true });
  const sendBtn = el("button", { class: "btn btn-primary", text: "Send" });
  const stopBtn = el("button", { class: "btn", text: "Stop", hidden: true });
  const attachBtn = el("button", {
    class: "btn btn-ghost btn-sm",
    text: "Attach image",
    onclick: () => fileInput.click(),
  });

  const composerBox = el("div", { class: "composer-box" }, [attachmentBar, textarea,
    el("div", { class: "composer-actions" }, [
      attachBtn,
      el("button", {
        class: "btn btn-ghost btn-sm",
        text: "Clear chat",
        onclick: () => { history.length = 0; paintLog(); },
      }),
      el("span", { class: "composer-hint", id: "ctx-hint" }),
      stopBtn,
      sendBtn,
    ]),
  ]);

  const metrics = el("div", { class: "live-metrics" });
  const modelSelect = el("select", {
    onchange: (event) => {
      current = candidates.find((m) => m.id === event.target.value);
      settings.model = current.id;
      saveSettings(settings);
      paintSidebar();
      paintHint();
    },
  }, candidates.map((m) =>
    el("option", { value: m.id, selected: m.id === current.id },
      `${m.name}${m.installed ? "" : "  (not downloaded)"}`),
  ));

  const systemInput = el("textarea", {
    rows: 3,
    placeholder: "Optional system prompt",
    oninput: (event) => { settings.system = event.target.value; saveSettings(settings); },
  });
  systemInput.value = settings.system;

  const tempInput = el("input", {
    type: "range", min: "0", max: "1.5", step: "0.05", value: String(settings.temperature),
    oninput: (event) => {
      settings.temperature = Number(event.target.value);
      $("#temp-value").textContent = settings.temperature.toFixed(2);
      saveSettings(settings);
    },
  });
  const maxTokensInput = el("input", {
    type: "number", min: "1", max: "8192", value: String(settings.maxTokens),
    oninput: (event) => {
      settings.maxTokens = Math.max(1, Number(event.target.value) || 512);
      saveSettings(settings);
    },
  });

  const modelInfo = el("div", { class: "col small" });

  const side = el("aside", { class: "chat-side" }, [
    el("div", {}, [el("h4", { text: "Model" }), modelSelect, modelInfo]),
    el("div", {}, [el("h4", { text: "This turn" }), metrics]),
    el("div", {}, [el("h4", { text: "System prompt" }), systemInput]),
    el("div", {}, [
      el("h4", { text: "Sampling" }),
      el("label", { class: "field" }, [
        el("div", { class: "row" }, [
          el("span", { text: "Temperature" }),
          el("span", { class: "grow" }),
          el("span", { id: "temp-value", class: "mono", text: settings.temperature.toFixed(2) }),
        ]),
        tempInput,
      ]),
      el("label", { class: "field" }, ["Max new tokens", maxTokensInput]),
    ]),
  ]);

  root.append(
    el("div", { class: "chat" }, [
      el("div", { class: "chat-main" }, [
        log,
        el("div", { class: "composer" }, [composerBox, fileInput]),
      ]),
      side,
    ]),
  );

  // --- painting ---

  function paintHint() {
    $("#ctx-hint").textContent =
      `${current.quant_label} · ${(current.context?.max || 0).toLocaleString()} ctx`;
    attachBtn.hidden = !current.capabilities?.vision;
  }

  function paintSidebar() {
    const bench = current.benchmark;
    const published = bench ? bench.tps?.["2048"] : null;
    fill(modelInfo, 
      el("div", { class: "row wrap", style: "margin-top:10px" }, [
        el("span", { class: `tag tag-${current.tier}`, text: current.tier.toUpperCase() }),
        el("span", { class: "tag", text: current.quant_label }),
        current.capabilities?.vision ? el("span", { class: "tag tag-vlm", text: "vision" }) : null,
        current.capabilities?.tools ? el("span", { class: "tag", text: "tools" }) : null,
      ]),
      el("div", { class: "muted", style: "margin-top:8px", text: current.description || "" }),
      published
        ? el("div", { class: "faint", style: "margin-top:8px" },
            `Certified ${fmtNumber(published)} tok/s at 2k context`)
        : null,
      current.installed
        ? null
        : el("div", { class: "notice", style: "margin-top:10px" }, [
            el("strong", { text: "Not downloaded. " }),
            "Pull it from the Models tab before running on the GPU engine.",
          ]),
    );
    paintHint();
  }

  function paintMetrics(usage, streaming = false) {
    const cells = [
      ["TTFT", usage?.ttft_s !== undefined ? fmtDuration(usage.ttft_s) : "-"],
      ["Tokens/s", usage?.tps ? fmtNumber(usage.tps) : "-"],
      ["Generated", usage?.completion_tokens ?? "-"],
      ["Prompt", usage?.prompt_tokens ?? "-"],
    ];
    fill(metrics, 
      ...cells.map(([k, v]) =>
        el("div", { class: "m" }, [el("div", { class: "k", text: k }), el("div", { class: "v", text: String(v) })]),
      ),
    );
    if (streaming) metrics.firstChild.querySelector(".v").classList.add("mono");
  }

  function messageNode(message, index) {
    const body = el("div", { class: "body" });
    if (message.images?.length) {
      body.append(
        el("div", { class: "imgs" }, message.images.map((src) => el("img", { src }))),
      );
    }
    const md = el("div", { class: "md" });
    renderMarkdown(md, message.content);
    body.append(md);

    const meta = el("div", { class: "meta" });
    if (message.usage) {
      meta.append(
        el("span", { text: `TTFT ${fmtDuration(message.usage.ttft_s)}` }),
        el("span", { text: `${fmtNumber(message.usage.tps)} tok/s` }),
        el("span", { text: `${message.usage.completion_tokens} tokens` }),
      );
    }
    if (message.role === "assistant" && message.content) {
      meta.append(copyButton(() => message.content, "Copy"));
    }
    if (meta.childElementCount) body.append(meta);

    return el("div", { class: `msg ${message.role}`, "data-index": index }, [
      el("div", { class: "who", text: message.role === "user" ? "YOU" : "EP" }),
      body,
    ]);
  }

  function paintLog() {
    clear(log);
    if (!history.length) {
      log.append(
        empty(
          vision ? "Ask about an image" : "Start a conversation",
          vision
            ? "Drop an image into the box below, or paste one from the clipboard, then ask a question about it."
            : "Pick a model on the right and send a message. Time to first token and decode rate are measured for every turn.",
        ),
      );
      paintMetrics(null);
      return;
    }
    history.forEach((message, index) => log.append(messageNode(message, index)));
    log.scrollTop = log.scrollHeight;
  }

  // --- sending ---

  function requestMessages() {
    const messages = [];
    if (settings.system.trim()) {
      messages.push({ role: "system", content: settings.system.trim() });
    }
    for (const message of history) {
      if (message.images?.length) {
        messages.push({
          role: message.role,
          content: [
            { type: "text", text: message.content },
            ...message.images.map((url) => ({ type: "image_url", image_url: { url } })),
          ],
        });
      } else {
        messages.push({ role: message.role, content: message.content });
      }
    }
    return messages;
  }

  async function send() {
    const text = textarea.value.trim();
    if (!text && !attachments.length) return;
    if (active) return;

    history.push({ role: "user", content: text, images: [...attachments] });
    textarea.value = "";
    textarea.style.height = "auto";
    attachments = [];
    paintAttachments();
    paintLog();

    const reply = { role: "assistant", content: "", images: [] };
    history.push(reply);
    const node = messageNode(reply, history.length - 1);
    log.append(node);
    const md = node.querySelector(".md");
    const caret = el("span", { class: "cursor-blink" });
    md.append(caret);
    log.scrollTop = log.scrollHeight;

    sendBtn.hidden = true;
    stopBtn.hidden = false;

    const started = performance.now();
    let tokens = 0;
    let firstAt = null;
    const atBottom = () => log.scrollHeight - log.scrollTop - log.clientHeight < 80;

    active = streamChat({
      model: current.id,
      messages: requestMessages().slice(0, -1),
      params: { temperature: settings.temperature, max_tokens: settings.maxTokens },
      onDelta: (delta) => {
        if (firstAt === null) firstAt = performance.now();
        tokens += 1;
        reply.content += delta;
        const stick = atBottom();
        renderMarkdown(md, reply.content);
        md.append(caret);
        if (stick) log.scrollTop = log.scrollHeight;
        const decode = (performance.now() - firstAt) / 1000;
        paintMetrics(
          {
            ttft_s: (firstAt - started) / 1000,
            tps: decode > 0 ? tokens / decode : 0,
            completion_tokens: tokens,
          },
          true,
        );
      },
      onDone: (usage, aborted) => {
        active = null;
        sendBtn.hidden = false;
        stopBtn.hidden = true;
        caret.remove();
        reply.usage = usage || {
          ttft_s: firstAt ? (firstAt - started) / 1000 : 0,
          tps: 0,
          completion_tokens: tokens,
          prompt_tokens: 0,
        };
        if (aborted) reply.content += "\n\n*(stopped)*";
        paintLog();
        paintMetrics(reply.usage);
      },
      onError: (error) => {
        active = null;
        sendBtn.hidden = false;
        stopBtn.hidden = true;
        caret.remove();
        history.pop();
        paintLog();
        toast(error.message, "error", 8000);
      },
    });
  }

  sendBtn.addEventListener("click", send);
  stopBtn.addEventListener("click", () => active?.abort());

  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 210)}px`;
  });
  textarea.addEventListener("focus", () => composerBox.classList.add("focus"));
  textarea.addEventListener("blur", () => composerBox.classList.remove("focus"));

  // --- images ---

  function paintAttachments() {
    attachmentBar.hidden = attachments.length === 0;
    fill(attachmentBar, 
      ...attachments.map((src, index) =>
        el("div", { class: "attachment" }, [
          el("img", { src }),
          el("button", {
            html: "&times;",
            title: "Remove",
            onclick: () => { attachments.splice(index, 1); paintAttachments(); },
          }),
        ]),
      ),
    );
  }

  async function addFiles(files) {
    const images = [...files].filter((f) => f.type.startsWith("image/"));
    if (!images.length) return;
    if (!current.capabilities?.vision) {
      const vlm = candidates.find((m) => m.capabilities?.vision) ||
        state.catalog.find((m) => m.capabilities?.vision);
      if (!vlm) {
        toast("No vision model is available in the catalog", "error");
        return;
      }
      // Switching for the user is friendlier than refusing the drop, but say so.
      current = vlm;
      modelSelect.value = vlm.id;
      settings.model = vlm.id;
      saveSettings(settings);
      paintSidebar();
      toast(`Switched to ${vlm.name} to handle the image`, "info");
    }
    for (const file of images) {
      try {
        attachments.push(await fileToDataURL(file));
      } catch (error) {
        toast(error.message, "error");
      }
    }
    paintAttachments();
  }

  fileInput.addEventListener("change", () => { addFiles(fileInput.files); fileInput.value = ""; });

  composerBox.addEventListener("dragover", (event) => {
    event.preventDefault();
    composerBox.classList.add("drag");
  });
  composerBox.addEventListener("dragleave", () => composerBox.classList.remove("drag"));
  composerBox.addEventListener("drop", (event) => {
    event.preventDefault();
    composerBox.classList.remove("drag");
    addFiles(event.dataTransfer.files);
  });
  const onPaste = (event) => {
    const files = [...(event.clipboardData?.files || [])];
    if (files.length) { event.preventDefault(); addFiles(files); }
  };
  textarea.addEventListener("paste", onPaste);

  paintSidebar();
  paintLog();
  textarea.focus();

  return () => active?.abort();
}
