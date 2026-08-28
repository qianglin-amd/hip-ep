<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Export recipes

A recipe is the reproducible half of the model zoo. Every certified model names the
recipe that produced it, and anyone porting their own weights picks one as a starting
point. Recipes are plain YAML: copy one, change `applies_to`, and run it.

```bash
# List what is available, including through the server
hipep recipes
curl -s http://127.0.0.1:11434/v1/recipes | jq '.recipes[] | {id, name, recommended_for}'

# Port a model with a named recipe
hipep port Qwen/Qwen2.5-7B-Instruct --recipe qwen-rtn-int4
```

## Choosing one

| If your model is | Start with | Why |
|---|---|---|
| A dense decoder and you want the simplest path | `int4-rtn-g128` | Round-to-nearest needs no calibration data, so it is fast and always reproducible |
| A dense decoder and you want the best quality at INT4 | `int4-awq-g128` | AWQ recovers most of the accuracy RTN loses, at the cost of a calibration pass |
| Llama 3.x | `llama-awq-int4-asym` | Asymmetric INT4 measurably helps this family |
| Mistral / Mixtral dense | `mistral-awq-int4` | Symmetric AWQ, the certified configuration |
| Qwen 2.5 / 3.x dense | `qwen-rtn-int4` | This family is unusually robust to RTN |
| Phi-4 | `phi-rtn-int4-block32` | Block 32 offsets this family's higher quantization error |
| A mixture-of-experts model | `moe-rtn-int4-block32` or `moe-awq-uint4-pergroup` | Expert weights need finer granularity than a dense tower |
| A vision-language model | the matching `*-vlm-*` recipe | Keeps the vision encoder in fp16, which is what protects image quality |
| Whisper | `whisper-fp16` | The encoder-decoder pair is small; quantization buys little and costs accuracy |
| A classifier or detector | `vision-fp16` | These models are activation bound, not weight bound |

## Schema

```yaml
id:              # stable identifier, matches the file name
name:            # human label shown in the console
description:     # one paragraph on what this recipe is for
recommended_for: # short phrase shown next to the name in pickers
applies_to:      # families and architectures the recipe was validated against
quantization:    # the plan itself; mirrors the catalog's quantization block
calibration:     # dataset and sample count, omitted for data-free schemes
export:          # how the ONNX graph is produced
build:           # how the OGA artifact is assembled
validation:      # what must hold before the result is publishable
```

## Project defaults

Two settings are applied by default and are deliberately not left to the caller:

- **`chunk_size: 512`** — Quark processes the model in chunks. 512 is the value the
  certified artifacts were produced with, and it keeps peak host memory bounded on a
  single-GPU machine.
- **`prune_logit: true`** — the language-model head is excluded from quantization and
  pruned to the tokens the model can actually emit. It is a large fraction of the
  parameters in a small model, and quantizing it hurts more than anything else in the
  graph.

Override them only with a measured reason, and record that reason in the recipe.

## Reproducing a certified model

Every certified entry in `modelzoo/catalog/` carries both the recipe name and the
upstream `base_model`, so a certified artifact can be rebuilt from source:

```bash
hipep port meta-llama/Llama-3.1-8B-Instruct \
  --recipe llama-awq-int4-asym \
  --benchmark
```

The benchmark stage then compares the rebuilt model against the published numbers for
the certified one. A rebuild that lands outside a few percent means the recipe and the
published artifact have drifted apart, which is a bug in one of them.
