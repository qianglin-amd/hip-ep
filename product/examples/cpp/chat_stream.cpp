// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Stream a chat completion from oga-server in C++.
//
//   cmake -B build -S . && cmake --build build
//   ./build/chat_stream --model llama3.1:8b --prompt "Write a haiku about GPUs."
//
// The server speaks the OpenAI chat API, so this is one POST and a line-oriented read
// of the response. There is no SDK and, apart from the platform sockets used by
// http_stream.hpp, no dependency: an application can drop these two files in and
// replace the transport with whatever HTTP client it already links.

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "http_stream.hpp"

namespace {

struct Options {
  std::string base = "http://127.0.0.1:11434";
  std::string model = "llama3.1:8b";
  std::string prompt = "Write a haiku about GPUs.";
  int max_tokens = 200;
};

Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const bool has_value = i + 1 < argc;
    if (flag == "--base" && has_value) {
      options.base = argv[++i];
    } else if (flag == "--model" && has_value) {
      options.model = argv[++i];
    } else if (flag == "--prompt" && has_value) {
      options.prompt = argv[++i];
    } else if (flag == "--max-tokens" && has_value) {
      options.max_tokens = std::atoi(argv[++i]);
    } else {
      std::cerr << "usage: chat_stream [--base URL] [--model ID] [--prompt TEXT]"
                   " [--max-tokens N]\n";
      std::exit(flag == "--help" ? 0 : 2);
    }
  }
  return options;
}

/// Escape a string for embedding in JSON.
std::string json_escape(const std::string &text) {
  std::string out;
  out.reserve(text.size() + 16);
  for (const char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char escaped[7];
          std::snprintf(escaped, sizeof(escaped), "\\u%04x", c);
          out += escaped;
        } else {
          out += c;
        }
    }
  }
  return out;
}

/// Pull one string value out of a flat JSON object.
///
/// A sample should not carry a JSON parser, and the two fields this needs -- the delta
/// content and the finish reason -- are plain strings. Use a real parser in an
/// application; the response also contains nested objects this cannot see.
bool json_string_field(const std::string &json, const std::string &key, std::string *out) {
  const std::string needle = "\"" + key + "\":\"";
  const size_t start = json.find(needle);
  if (start == std::string::npos) {
    return false;
  }
  out->clear();
  for (size_t i = start + needle.size(); i < json.size(); ++i) {
    const char c = json[i];
    if (c == '"') {
      return true;
    }
    if (c != '\\' || i + 1 >= json.size()) {
      *out += c;
      continue;
    }
    switch (json[++i]) {
      case 'n': *out += '\n'; break;
      case 'r': *out += '\r'; break;
      case 't': *out += '\t'; break;
      case 'u': {
        // Unicode escapes are passed through as '?' rather than decoded: a sample that
        // hand-rolls UTF-8 conversion teaches the wrong lesson.
        i += 4;
        *out += '?';
        break;
      }
      default: *out += json[i];
    }
  }
  return true;
}

double json_number_field(const std::string &json, const std::string &key) {
  const std::string needle = "\"" + key + "\":";
  const size_t start = json.find(needle);
  if (start == std::string::npos) {
    return -1.0;
  }
  return std::atof(json.c_str() + start + needle.size());
}

}  // namespace

int main(int argc, char **argv) {
  const Options options = parse_args(argc, argv);

  const std::string body = std::string("{") + "\"model\":\"" + json_escape(options.model) +
                           "\"," + "\"messages\":[{\"role\":\"user\",\"content\":\"" +
                           json_escape(options.prompt) + "\"}]," + "\"max_tokens\":" +
                           std::to_string(options.max_tokens) + "," + "\"stream\":true}";

  using clock = std::chrono::steady_clock;
  const clock::time_point started = clock::now();
  clock::time_point first_token;
  bool seen_token = false;
  int tokens = 0;
  double server_ttft = -1.0;
  double server_tps = -1.0;

  try {
    const hipep::Session session;
    const int status = session.post_lines(
        options.base, "/v1/chat/completions", body, [&](const std::string &line) {
          // Server-sent events arrive as "data: {json}" lines separated by blank lines.
          if (line.rfind("data: ", 0) != 0) {
            return;
          }
          const std::string payload = line.substr(6);
          if (payload == "[DONE]" || payload == "\"[DONE]\"") {
            return;
          }

          std::string content;
          if (json_string_field(payload, "content", &content) && !content.empty()) {
            if (!seen_token) {
              first_token = clock::now();
              seen_token = true;
            }
            ++tokens;
            std::cout << content << std::flush;
          }
          // The final frame carries no delta, only usage. HIP EP adds ttft_s and tps.
          if (payload.find("\"usage\"") != std::string::npos) {
            server_ttft = json_number_field(payload, "ttft_s");
            server_tps = json_number_field(payload, "tps");
          }
        });

    std::cout << "\n";
    if (status != 200) {
      std::cerr << "request failed with HTTP " << status << "\n";
      return 1;
    }
  } catch (const std::exception &error) {
    std::cerr << "\nerror: " << error.what() << "\n";
    return 1;
  }

  const auto seconds = [](clock::duration d) {
    return std::chrono::duration<double>(d).count();
  };
  std::fprintf(stderr, "\n%d chunks in %.2fs", tokens, seconds(clock::now() - started));
  if (seen_token) {
    std::fprintf(stderr, ", client TTFT %.3fs", seconds(first_token - started));
  }
  if (server_ttft >= 0.0 && server_tps >= 0.0) {
    std::fprintf(stderr, ", server TTFT %.3fs at %.1f tok/s", server_ttft, server_tps);
  }
  std::fprintf(stderr, "\n");
  return 0;
}
