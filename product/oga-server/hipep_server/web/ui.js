// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// DOM helpers, formatters, and a small Markdown renderer.
//
// The Markdown renderer is hand-written rather than pulled from a CDN so the console
// keeps working offline and inside air-gapped labs. It escapes first and only then
// applies inline formatting, so model output can never inject markup.

/* ---------- DOM ---------- */

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, value);
  }
  return add(node, children);
}

/**
 * Append children, skipping the empty ones.
 *
 * Always use this rather than `node.append(...)`: the DOM method stringifies null,
 * so a conditional child written as `cond ? node : null` renders the word "null".
 */
export function add(node, ...children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
  return node;
}

/** Empty a node and append fresh children in one call. */
export function fill(node, ...children) {
  return add(clear(node), ...children);
}

/* ---------- formatters ---------- */

export function fmtBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1; }
  return `${i === 0 ? value : value.toFixed(1)} ${units[i]}`;
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "-";
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m < 60) return `${m}m ${s}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function fmtNumber(value, digits = 1) {
  if (value === null || value === undefined) return "-";
  return Number(value).toFixed(digits);
}

export function fmtContext(tokens) {
  if (!tokens) return "-";
  if (tokens >= 1024 && tokens % 1024 === 0) return `${tokens / 1024}k`;
  if (tokens >= 1000) return `${Math.round(tokens / 1000)}k`;
  return String(tokens);
}

export function timeAgo(timestamp) {
  const seconds = Date.now() / 1000 - timestamp;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** Percent change rendered with the right sign colour for the metric's direction. */
export function delta(current, previous, { lowerIsBetter = false } = {}) {
  if (current === null || previous === null || current === undefined || previous === undefined) {
    return null;
  }
  if (!previous) return null;
  const pct = ((current - previous) / previous) * 100;
  const better = lowerIsBetter ? pct < 0 : pct > 0;
  return el("span", {
    class: Math.abs(pct) < 0.05 ? "faint" : better ? "delta-up" : "delta-down",
    text: `${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%`,
  });
}

/* ---------- feedback ---------- */

export function toast(message, kind = "info", ms = 4200) {
  const node = el("div", { class: `toast ${kind}`, text: message });
  $("#toast-root").append(node);
  setTimeout(() => {
    node.style.opacity = "0";
    node.style.transition = "opacity .2s";
    setTimeout(() => node.remove(), 220);
  }, ms);
}

export function modal(title, body, { subtitle } = {}) {
  const backdrop = el("div", { class: "modal-backdrop" });
  const close = () => {
    backdrop.remove();
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (event) => { if (event.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);

  backdrop.append(
    el("div", { class: "modal", onclick: (e) => e.stopPropagation() }, [
      el("div", { class: "modal-head" }, [
        el("div", {}, [
          el("h2", { text: title }),
          subtitle ? el("div", { class: "muted small", text: subtitle }) : null,
        ]),
        el("button", { class: "btn btn-ghost", onclick: close, html: "&times;" }),
      ]),
      el("div", { class: "modal-body" }, body),
    ]),
  );
  backdrop.addEventListener("click", close);
  $("#modal-root").append(backdrop);
  return { close };
}

export function copyButton(getText, label = "Copy") {
  return el("button", {
    class: "btn btn-sm btn-ghost copy",
    text: label,
    onclick: async (event) => {
      try {
        await navigator.clipboard.writeText(getText());
        const button = event.currentTarget;
        button.textContent = "Copied";
        setTimeout(() => { button.textContent = label; }, 1400);
      } catch {
        toast("Clipboard is blocked by the browser", "error");
      }
    },
  });
}

export function empty(title, detail, action) {
  return el("div", { class: "empty" }, [
    el("h3", { text: title }),
    el("div", { text: detail }),
    action ? el("div", { style: "margin-top:16px" }, action) : null,
  ]);
}

/* ---------- markdown ---------- */

const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
export const escapeHTML = (text) => String(text).replace(/[&<>"']/g, (c) => ESCAPES[c]);

function inline(text) {
  let out = escapeHTML(text);
  // Code spans first: their contents must not be reinterpreted as emphasis or links.
  const codes = [];
  out = out.replace(/`([^`]+)`/g, (_, code) => `\u0000${codes.push(code) - 1}\u0000`);
  out = out
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
      (_, label, href) => `<a href="${href}" target="_blank" rel="noopener">${label}</a>`)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|\W)_([^_\n]+)_(?=\W|$)/g, "$1<em>$2</em>")
    .replace(/~~([^~]+)~~/g, "<del>$1</del>");
  return out.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[i]}</code>`);
}

/**
 * Render a useful subset of Markdown: fenced code, headings, lists, tables, block
 * quotes, rules and inline formatting. Streaming-safe — an unterminated fence renders
 * as an open code block rather than swallowing the rest of the document.
 */
export function markdown(source) {
  const lines = String(source ?? "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;

  const isTableRule = (line) => /^\s*\|?[\s:-]*-[\s|:-]*\|?\s*$/.test(line) && line.includes("-");

  while (i < lines.length) {
    const line = lines[i];

    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const body = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) body.push(lines[i++]);
      i += 1;
      out.push(
        `<pre data-lang="${escapeHTML(fence[1])}"><code>${escapeHTML(body.join("\n"))}</code></pre>`,
      );
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      const level = Math.min(heading[1].length, 4);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { out.push("<hr>"); i += 1; continue; }

    if (line.includes("|") && isTableRule(lines[i + 1] || "")) {
      const cells = (row) =>
        row.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map((c) => c.trim());
      const header = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
        rows.push(cells(lines[i++]));
      }
      out.push(
        "<table><thead><tr>" +
          header.map((c) => `<th>${inline(c)}</th>`).join("") +
          "</tr></thead><tbody>" +
          rows
            .map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("") + "</tr>")
            .join("") +
          "</tbody></table>",
      );
      continue;
    }

    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || ordered) {
      const tag = bullet ? "ul" : "ol";
      const pattern = bullet ? /^\s*[-*+]\s+(.*)$/ : /^\s*\d+[.)]\s+(.*)$/;
      const items = [];
      while (i < lines.length) {
        const match = lines[i].match(pattern);
        if (!match) break;
        items.push(`<li>${inline(match[1])}</li>`);
        i += 1;
      }
      out.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoted = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoted.push(lines[i++].replace(/^\s*>\s?/, ""));
      }
      out.push(`<blockquote>${markdown(quoted.join("\n"))}</blockquote>`);
      continue;
    }

    if (!line.trim()) { i += 1; continue; }

    const paragraph = [];
    while (i < lines.length && lines[i].trim() && !/^(#{1,6}\s|```|\s*[-*+]\s|\s*\d+[.)]\s|\s*>)/.test(lines[i])) {
      paragraph.push(lines[i++]);
    }
    out.push(`<p>${inline(paragraph.join(" "))}</p>`);
  }

  return out.join("");
}

/** Render markdown into a node and attach copy buttons to code blocks. */
export function renderMarkdown(node, source) {
  node.innerHTML = markdown(source);
  for (const pre of $$("pre", node)) {
    if (pre.querySelector(".copy")) continue;
    pre.append(copyButton(() => pre.querySelector("code").textContent));
  }
  return node;
}
