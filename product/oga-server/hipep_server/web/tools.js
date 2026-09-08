// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// The tools the agent view can actually run.
//
// Every one of these executes for real: the clock reads the clock, the calculator does
// the arithmetic, and the two server tools call this server's own API. A demo whose
// tools return canned strings proves that the model emits a tool call and nothing about
// whether the round trip works, which is the part that breaks.
//
// Each entry pairs the JSON schema sent to the model with the function that runs when
// the model asks for it. Handlers are async and return a value that is JSON-encoded
// into the tool result message.

import { api } from "./api.js";

/**
 * Evaluate arithmetic without handing the model's text to eval().
 *
 * A tool argument is model output, so it is untrusted input in the ordinary sense: the
 * model can be talked into emitting whatever the user's prompt asks for. This is a
 * recursive-descent parser over numbers and the five operators, so a string that is not
 * arithmetic is a parse error rather than something that runs.
 */
export function evaluateArithmetic(expression) {
  const text = String(expression);

  // Checked before tokenizing, because a tokenizer that matches what it recognises
  // silently drops everything else: "globalThis.alert(1)" tokenizes to "(", "1", ")"
  // and evaluates to 1. Nothing is executed either way, but a calculator that returns
  // a confident answer for input that is not arithmetic is worse than one that
  // refuses, since the model repeats the number as fact.
  const unsupported = text.replace(/[\d.\s+\-*/%()]/g, "");
  if (unsupported) {
    throw new Error(`not arithmetic: unsupported characters ${JSON.stringify(unsupported)}`);
  }

  const tokens = text.match(/\d+\.?\d*|[+\-*/%()]|\s+/g) || [];
  const stream = tokens.filter((t) => !/^\s+$/.test(t));
  let at = 0;

  const peek = () => stream[at];
  const take = (token) => {
    if (peek() !== token) throw new Error(`expected ${token}`);
    at += 1;
  };

  // expression := term (('+' | '-') term)*
  const parseExpression = () => {
    let value = parseTerm();
    while (peek() === "+" || peek() === "-") {
      const operator = stream[at++];
      const right = parseTerm();
      value = operator === "+" ? value + right : value - right;
    }
    return value;
  };

  // term := factor (('*' | '/' | '%') factor)*
  const parseTerm = () => {
    let value = parseFactor();
    while (peek() === "*" || peek() === "/" || peek() === "%") {
      const operator = stream[at++];
      const right = parseFactor();
      if ((operator === "/" || operator === "%") && right === 0) {
        throw new Error("division by zero");
      }
      if (operator === "*") value *= right;
      else if (operator === "/") value /= right;
      else value %= right;
    }
    return value;
  };

  // factor := '-'? (number | '(' expression ')')
  const parseFactor = () => {
    if (peek() === "-") {
      at += 1;
      return -parseFactor();
    }
    if (peek() === "(") {
      take("(");
      const value = parseExpression();
      take(")");
      return value;
    }
    const token = stream[at++];
    if (token === undefined || !/^\d/.test(token)) {
      throw new Error(`not arithmetic: ${expression}`);
    }
    return Number(token);
  };

  const value = parseExpression();
  if (at !== stream.length) throw new Error(`not arithmetic: ${expression}`);
  if (!Number.isFinite(value)) throw new Error("the result is not a finite number");
  return value;
}

export const TOOLS = [
  {
    name: "get_current_time",
    schema: {
      type: "function",
      function: {
        name: "get_current_time",
        description:
          "Get the current date and time on the machine running this browser. " +
          "Use this whenever the answer depends on what time it is now.",
        parameters: {
          type: "object",
          properties: {
            timezone: {
              type: "string",
              description:
                "An IANA timezone such as Europe/London. Omit it for this machine's " +
                "own timezone.",
            },
          },
          required: [],
        },
      },
    },
    async run({ timezone } = {}) {
      const now = new Date();
      const here = Intl.DateTimeFormat().resolvedOptions().timeZone;

      // Models pass words like "local", "system" or "current" here, partly because
      // asking for the current time invites them and partly because the description
      // used to say "defaults to local". Only IANA names work, so an unusable value
      // falls back to this machine's zone and says which one it used. Returning an
      // error instead spends a whole round trip teaching the model what a timezone is,
      // and small models do not learn it.
      let zone = timezone;
      if (zone) {
        try {
          now.toLocaleString(undefined, { timeZone: zone });
        } catch {
          zone = null;
        }
      }

      return {
        iso: now.toISOString(),
        local: now.toLocaleString(undefined, zone ? { timeZone: zone } : {}),
        timezone: zone || here,
        ...(timezone && !zone
          ? { note: `"${timezone}" is not an IANA timezone; used ${here} instead` }
          : {}),
      };
    },
  },
  {
    name: "calculate",
    schema: {
      type: "function",
      function: {
        name: "calculate",
        description:
          "Evaluate an arithmetic expression with + - * / % and parentheses. " +
          "Use this instead of doing arithmetic yourself.",
        parameters: {
          type: "object",
          properties: {
            expression: {
              type: "string",
              description: "The expression to evaluate, for example (2500 * 12) / 7",
            },
          },
          required: ["expression"],
        },
      },
    },
    async run({ expression } = {}) {
      return { expression, result: evaluateArithmetic(expression) };
    },
  },
  {
    name: "get_gpu_status",
    schema: {
      type: "function",
      function: {
        name: "get_gpu_status",
        description:
          "Get the GPU, engine and current load of the HIP EP server answering this " +
          "conversation. Use this for questions about the hardware or the server.",
        parameters: { type: "object", properties: {}, required: [] },
      },
    },
    async run() {
      const status = await api.status();
      const gpu = status.host?.gpus?.[0];
      return {
        gpu: gpu ? `${gpu.profile_name} (${gpu.gfx_arch})` : "no AMD GPU detected",
        gpu_memory_gb: gpu?.memory_gb ?? null,
        engine: status.engine?.name,
        engine_is_real: status.engine?.real,
        loaded_models: status.load?.loaded_models ?? null,
        active_requests: status.load?.active_requests ?? null,
        sessions: status.load?.sessions ?? null,
      };
    },
  },
  {
    name: "list_models",
    schema: {
      type: "function",
      function: {
        name: "list_models",
        description:
          "List the models this HIP EP server can serve, with their quantization. " +
          "Use this for questions about which models are available.",
        parameters: {
          type: "object",
          properties: {
            tier: {
              type: "string",
              description: "Restrict to one tier, such as llm or vlm.",
            },
          },
          required: [],
        },
      },
    },
    async run({ tier } = {}) {
      const catalog = await api.catalog(tier ? { tier } : {});
      return {
        count: catalog.models.length,
        models: catalog.models.slice(0, 20).map((m) => ({
          id: m.id,
          quantization: m.quantization,
          context_length: m.context_length,
        })),
      };
    },
  },
];

export const TOOLS_BY_NAME = new Map(TOOLS.map((tool) => [tool.name, tool]));

/**
 * Run one tool call and return the string that goes back to the model.
 *
 * A tool that throws returns its error rather than aborting the conversation: a model
 * that asked for a timezone that does not exist can recover if it is told so, and
 * cannot if the loop stops.
 */
export async function runToolCall(call, extraTools = new Map()) {
  const name = call.function?.name;
  const tool = extraTools.get(name) || TOOLS_BY_NAME.get(name);
  if (!tool) return JSON.stringify({ error: `no tool named ${name} is available here` });

  let args = {};
  try {
    args = call.function?.arguments ? JSON.parse(call.function.arguments) : {};
  } catch {
    // Small models emit not-quite-JSON arguments often enough that failing the call
    // outright would misattribute a formatting slip to the tool.
    return JSON.stringify({
      error: `arguments were not valid JSON: ${call.function?.arguments ?? ""}`,
    });
  }

  try {
    return JSON.stringify(await tool.run(args));
  } catch (error) {
    return JSON.stringify({ error: String(error.message || error) });
  }
}
