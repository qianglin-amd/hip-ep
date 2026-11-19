/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// oga-chat: generate text with an ONNX Runtime GenAI (OGA) model package on
// the HIP EP plugin, without Python. The model is any directory holding a
// genai_config.json; whatever provider the config names, the model runs on
// the HIP EP unless --cpu is given.

#include "ort_genai.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

constexpr const char *kUsage = R"(usage: oga-chat <model_dir> [options]

Runs an ONNX Runtime GenAI model on the HIP EP. Without -p it reads one
prompt per line and answers each; an empty line or end of input exits.

  -p, --prompt TEXT     answer TEXT and exit
  -n, --max-new N       stop after N generated tokens (default 512)
  --max-length N        KV cache length, prompt plus generated tokens
                        (default: prompt length + --max-new)
  --system TEXT         system message for the chat template
  --no-template         feed the prompt as is, without the chat template
  --cpu                 run on the CPU instead of the HIP EP
  --ep-lib PATH         HIP EP library (default: hipgpu.dll next to this exe)
  --ep-name NAME        EP registration name (default AMDGPU)
  -q, --quiet           print only the generated text
)";

struct Options {
  std::string modelDir;
  std::string prompt;
  std::string system;
  std::string epLib;
  std::string epName = "AMDGPU";
  // Contents of <model_dir>/chat_template.jinja; some exports keep the
  // template there instead of in tokenizer_config.json.
  std::string chatTemplate;
  int maxNew = 512;
  int maxLength = 0;
  bool cpu = false;
  bool noTemplate = false;
  bool quiet = false;
};

std::filesystem::path exeDirectory() {
#ifdef _WIN32
  wchar_t buf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH);
  return std::filesystem::path(std::wstring(buf, n)).parent_path();
#else
  return std::filesystem::canonical("/proc/self/exe").parent_path();
#endif
}

std::string jsonEscape(const std::string &s) {
  std::string out;
  for (const char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "\\u%04x", c);
        out += hex;
      } else {
        out += c;
      }
    }
  }
  return out;
}

std::string buildPrompt(const OgaTokenizer &tokenizer, const Options &opt,
                        const std::string &user) {
  if (opt.noTemplate)
    return user;
  std::string messages = "[";
  if (!opt.system.empty())
    messages +=
        R"({"role": "system", "content": ")" + jsonEscape(opt.system) + "\"}, ";
  messages += R"({"role": "user", "content": ")" + jsonEscape(user) + "\"}]";
  try {
    OgaString text = tokenizer.ApplyChatTemplate(
        opt.chatTemplate.empty() ? nullptr : opt.chatTemplate.c_str(),
        messages.c_str(), nullptr, true);
    return std::string(static_cast<const char *>(text));
  } catch (const std::exception &) {
    // Models without a chat template take the raw prompt.
    return user;
  }
}

void answer(OgaModel &model, const OgaTokenizer &tokenizer, const Options &opt,
            const std::string &user) {
  const std::string text = buildPrompt(tokenizer, opt, user);
  auto sequences = OgaSequences::Create();
  tokenizer.Encode(text.c_str(), *sequences);
  const size_t promptTokens = sequences->SequenceCount(0);

  auto params = OgaGeneratorParams::Create(model);
  // The KV cache is sized for max_length up front. A model's configured
  // max_length is often its whole context window, which can exceed device
  // memory, so default to what this answer can use.
  const int maxLength = opt.maxLength > 0
                            ? opt.maxLength
                            : static_cast<int>(promptTokens) + opt.maxNew;
  params->SetSearchOption("max_length", maxLength);
  params->SetSearchOptionBool("do_sample", false);

  auto generator = OgaGenerator::Create(model, *params);
  auto stream = OgaTokenizerStream::Create(tokenizer);
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  generator->AppendTokenSequences(*sequences);

  int generated = 0;
  Clock::time_point firstToken;
  while (!generator->IsDone() && generated < opt.maxNew) {
    generator->GenerateNextToken();
    if (generated == 0)
      firstToken = Clock::now();
    const int32_t token = generator->GetNextTokens()[0];
    std::cout << stream->Decode(token) << std::flush;
    ++generated;
  }
  const auto end = Clock::now();
  std::cout << std::endl;
  if (opt.quiet || generated == 0)
    return;
  const double ttftMs =
      std::chrono::duration<double, std::milli>(firstToken - start).count();
  const double decodeS =
      std::chrono::duration<double>(end - firstToken).count();
  std::fprintf(stderr,
               "[prompt %zu tokens, first token %.0f ms, %d tokens, "
               "%.1f tokens/s]\n",
               promptTokens, ttftMs, generated,
               generated > 1 ? (generated - 1) / decodeS : 0.0);
}

bool parseArgs(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "oga-chat: %s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-h" || a == "--help") {
      std::fputs(kUsage, stdout);
      std::exit(0);
    } else if (a == "-p" || a == "--prompt") {
      opt.prompt = value();
    } else if (a == "-n" || a == "--max-new") {
      opt.maxNew = std::atoi(value());
    } else if (a == "--max-length") {
      opt.maxLength = std::atoi(value());
    } else if (a == "--system") {
      opt.system = value();
    } else if (a == "--no-template") {
      opt.noTemplate = true;
    } else if (a == "--cpu") {
      opt.cpu = true;
    } else if (a == "--ep-lib") {
      opt.epLib = value();
    } else if (a == "--ep-name") {
      opt.epName = value();
    } else if (a == "-q" || a == "--quiet") {
      opt.quiet = true;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "oga-chat: unknown option %s\n%s", a.c_str(),
                   kUsage);
      return false;
    } else if (opt.modelDir.empty()) {
      opt.modelDir = a;
    } else {
      std::fprintf(stderr, "oga-chat: unexpected argument %s\n", a.c_str());
      return false;
    }
  }
  if (opt.modelDir.empty()) {
    std::fputs(kUsage, stderr);
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt))
    return 2;

  OgaHandle handle;
  try {
    if (!opt.cpu) {
      std::filesystem::path lib = opt.epLib;
      if (lib.empty()) {
        if (const char *env = std::getenv("HIPEP_EP_LIB"); env && *env)
          lib = env;
        else
          lib = exeDirectory() / "hipgpu.dll";
      }
      if (!std::filesystem::exists(lib)) {
        std::fprintf(stderr, "oga-chat: HIP EP library not found: %s\n",
                     lib.string().c_str());
        return 1;
      }
      OgaRegisterExecutionProviderLibrary(opt.epName.c_str(),
                                          lib.string().c_str());
    }

    const auto loadStart = std::chrono::steady_clock::now();
    auto config = OgaConfig::Create(opt.modelDir.c_str());
    config->ClearProviders();
    if (!opt.cpu)
      config->AppendProvider(opt.epName.c_str());
    auto model = OgaModel::Create(*config);
    auto tokenizer = OgaTokenizer::Create(*model);
    if (std::ifstream in(std::filesystem::path(opt.modelDir) /
                         "chat_template.jinja");
        in) {
      opt.chatTemplate.assign(std::istreambuf_iterator<char>(in), {});
    }
    if (!opt.quiet)
      std::fprintf(stderr, "[model loaded on %s in %.1f s]\n",
                   opt.cpu ? "CPU" : opt.epName.c_str(),
                   std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - loadStart)
                       .count());

    if (!opt.prompt.empty()) {
      answer(*model, *tokenizer, opt, opt.prompt);
      return 0;
    }
    std::string line;
    while (true) {
      std::cout << "> " << std::flush;
      if (!std::getline(std::cin, line) || line.empty())
        break;
      answer(*model, *tokenizer, opt, line);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "oga-chat: %s\n", e.what());
    return 1;
  }
  return 0;
}
