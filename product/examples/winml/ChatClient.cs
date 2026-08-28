// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// Stream a chat completion from oga-server in C# / .NET.
//
//   dotnet run -- --model llama3.1:8b --prompt "Summarise DirectML in one sentence."
//
// A Windows application already has HttpClient and System.Text.Json, so talking to
// oga-server needs no package reference at all. This is the same pattern a WinUI or
// Windows ML application uses: the model runs in the local server process on the AMD
// GPU, and the app stays a thin client that can be restarted without reloading weights.

using System.Diagnostics;
using System.Text;
using System.Text.Json;

internal static class ChatClient
{
    private static async Task<int> Main(string[] args)
    {
        var options = Options.Parse(args);
        if (options is null)
        {
            return 2;
        }

        using var http = new HttpClient { BaseAddress = new Uri(options.Base) };
        // Generation is slower than the default 100 s timeout allows at long contexts.
        http.Timeout = TimeSpan.FromMinutes(10);

        var request = new
        {
            model = options.Model,
            messages = new[] { new { role = "user", content = options.Prompt } },
            max_tokens = options.MaxTokens,
            stream = true,
        };

        var payload = new StringContent(
            JsonSerializer.Serialize(request), Encoding.UTF8, "application/json");

        var clock = Stopwatch.StartNew();
        TimeSpan? firstToken = null;
        var chunks = 0;
        double serverTtft = -1;
        double serverTps = -1;

        try
        {
            using var message = new HttpRequestMessage(HttpMethod.Post, "v1/chat/completions")
            {
                Content = payload,
            };

            // ResponseHeadersRead is what makes this stream. The default,
            // ResponseContentRead, buffers the whole body before returning, so every
            // token would arrive at once and the measured TTFT would be the total time.
            using var response = await http.SendAsync(
                message, HttpCompletionOption.ResponseHeadersRead);

            if (!response.IsSuccessStatusCode)
            {
                var detail = await response.Content.ReadAsStringAsync();
                await Console.Error.WriteLineAsync($"HTTP {(int)response.StatusCode}: {detail}");
                return 1;
            }

            using var stream = await response.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(stream);

            // Server-sent events arrive as "data: {json}" lines. Reading line by line is
            // what makes this a stream: ReadAsStringAsync would wait for the last token.
            while (await reader.ReadLineAsync() is { } line)
            {
                if (!line.StartsWith("data: ", StringComparison.Ordinal))
                {
                    continue;
                }

                var body = line["data: ".Length..];
                if (body is "[DONE]" or "\"[DONE]\"")
                {
                    break;
                }

                using var document = JsonDocument.Parse(body);
                var root = document.RootElement;

                if (root.TryGetProperty("choices", out var choices) && choices.GetArrayLength() > 0
                    && choices[0].GetProperty("delta").TryGetProperty("content", out var content))
                {
                    var text = content.GetString();
                    if (!string.IsNullOrEmpty(text))
                    {
                        firstToken ??= clock.Elapsed;
                        chunks++;
                        Console.Write(text);
                    }
                }

                // The final frame carries usage. ttft_s and tps are HIP EP extensions.
                if (root.TryGetProperty("usage", out var usage) &&
                    usage.ValueKind == JsonValueKind.Object)
                {
                    if (usage.TryGetProperty("ttft_s", out var ttft))
                    {
                        serverTtft = ttft.GetDouble();
                    }

                    if (usage.TryGetProperty("tps", out var tps))
                    {
                        serverTps = tps.GetDouble();
                    }
                }
            }
        }
        catch (HttpRequestException error)
        {
            await Console.Error.WriteLineAsync(
                $"cannot reach {options.Base}: {error.Message} (is oga-server running?)");
            return 1;
        }

        Console.WriteLine();

        var report = new StringBuilder($"{chunks} chunks in {clock.Elapsed.TotalSeconds:F2}s");
        if (firstToken is { } ttftClient)
        {
            report.Append($", client TTFT {ttftClient.TotalSeconds:F3}s");
        }

        if (serverTtft >= 0 && serverTps >= 0)
        {
            report.Append($", server TTFT {serverTtft:F3}s at {serverTps:F1} tok/s");
        }

        await Console.Error.WriteLineAsync(report.ToString());
        return 0;
    }

    private sealed record Options(string Base, string Model, string Prompt, int MaxTokens)
    {
        public static Options? Parse(string[] args)
        {
            var baseUrl = "http://127.0.0.1:11434";
            var model = "llama3.1:8b";
            var prompt = "Summarise DirectML in one sentence.";
            var maxTokens = 200;

            for (var i = 0; i < args.Length; i++)
            {
                var hasValue = i + 1 < args.Length;
                switch (args[i])
                {
                    case "--base" when hasValue:
                        baseUrl = args[++i];
                        break;
                    case "--model" when hasValue:
                        model = args[++i];
                        break;
                    case "--prompt" when hasValue:
                        prompt = args[++i];
                        break;
                    case "--max-tokens" when hasValue:
                        maxTokens = int.Parse(args[++i]);
                        break;
                    default:
                        Console.Error.WriteLine(
                            "usage: dotnet run -- [--base URL] [--model ID] [--prompt TEXT] " +
                            "[--max-tokens N]");
                        return null;
                }
            }

            return new Options(baseUrl.TrimEnd('/') + "/", model, prompt, maxTokens);
        }
    }
}
