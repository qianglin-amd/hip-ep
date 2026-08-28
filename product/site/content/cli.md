---
title: CLI mode
subtitle: Everything the console does, from a terminal
section: Instructions
order: 9
---

The `hipep` command is a client of the same HTTP API the console and your application
use. There is no privileged path: if a number or a behaviour differs between the CLI and
your code, that is a bug rather than a configuration difference.

```bash
hipep serve                     # start the server and the console
hipep models                    # the certified zoo
hipep pull llama3.1:8b          # download a model
hipep run llama3.1:8b "why?"    # generate
hipep status                    # engine, hardware, load, sessions
```

Every subcommand except `serve` takes `--url` and honours `$HIPEP_URL`, so the same
commands manage a server on another machine.

## Generating

```bash
# one-shot
hipep run llama3.1:8b "Explain quantization in two sentences."

# interactive; blank line or Ctrl-D exits
hipep run llama3.1:8b

# no model named: picks something already installed
hipep run

# an image, to a vision model
hipep run gemma3-vl:4b --image photo.jpg "What is in this picture?"

# from a pipe, into a pipe
cat error.log | hipep run --quiet -n 200 > summary.txt
```

After each turn `hipep run` prints the engine's TTFT and tokens per second alongside the
time the client itself saw the first token. The two differ by the HTTP and scheduling
cost, and seeing both is the difference between "the model is slow" and "my transport is
slow".

`--quiet` prints only generated text, which is what you want when the output is going
into a file or another program. Ctrl-C cancels a turn and keeps the conversation.

A model that is not installed is pulled first. Pass `--no-pull` in a script that should
fail rather than start a download.

## Managing models

```bash
hipep models                       # everything certified
hipep models --tier vlm            # vision models only
hipep models --search qwen         # by name, family or id
hipep show llama3.1:8b             # the model card, including quantization schema
hipep ls                           # installed here
hipep rm llama3.1:8b               # delete it and its files
hipep recipes                      # export and quantization recipes
```

`hipep models` marks what is installed and what will not fit this machine's memory, so a
model you cannot run is visible as such before you spend the download on it.

## Porting your own model

```bash
hipep port Qwen/Qwen2.5-7B-Instruct --name qwen2.5:7b-mine
hipep port ./my-checkpoint --recipe int4-awq-g128 --chunk-size 256
```

Defaults are `--recipe int4-rtn-g128`, `--chunk-size 512` and logit pruning on. The eight
stages stream to the terminal as they run; `-v` adds the job log. The model is usable the
moment the job finishes. See [Port a model](port-a-model.html).

## Sharing a port

A model you ported on one machine can be installed on another without repeating the
work:

```bash
# on the machine that has it, with the server reachable on the network
hipep serve --host 0.0.0.0

# on the other machine
hipep pull qwen2.5:7b-mine --from http://porting-box:11434
```

The transfer is per-file and resumable: re-running after an interruption fetches only
what is missing. The peer's quantization schema and capabilities travel with it, so the
imported model's card matches the original.

## Measuring

```bash
hipep bench llama3.1:8b
hipep bench llama3.1:8b --contexts 128 2048 16384 --repeats 5
```

The result table puts your numbers beside the certified ones, with HTTP cost in its own
column and the delta computed against the engine measurement. A run on the replay engine
reproduces the published timings and says so; it does not measure your machine.

## Watching a busy server

```bash
hipep status          # engine, GPU, limits, totals, every session
hipep ps              # requests in flight and models resident
hipep ps --watch      # the same, live
hipep jobs            # recent jobs
hipep jobs port-1a2b3c --follow --logs
hipep cancel port-1a2b3c
```

`hipep status` is the first thing to run when something is wrong: it names the engine and
*why* that engine was chosen, which answers the most common surprise on a new machine.

## Scripting

Every listing command takes `--json`, so nothing needs to be parsed out of a table:

```bash
# the fastest certified model that fits this machine
hipep models --json | jq -r '[.[] | select(.fits)]
  | max_by(.benchmark.tps["2048"] // 0) | .id'
```

Exit statuses are stable: `0` success, `1` the operation failed, `3` no server at that
address, `4` the server rejected the request, `130` interrupted.
