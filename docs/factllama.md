# FactLLaMA inference

Evidence-based claim verification with `meta-llama/Llama-3.2-3B-Instruct`
and the QLoRA adapter `factllama_final_adapter_v2`.

> **Status:** the inference scaffolding is implemented and unit-tested with
> mocks. The saved adapter has **not** been loaded or run yet, so real
> end-to-end inference is unverified. The 61.5% accuracy / ~0.606 macro-F1
> figures come from the fine-tuning evaluation and are not re-measured here.

## Layout

```
src/factllama/
  prompt_builder.py   system/user prompt + evidence validation
  output_parser.py    true | half | false parsing
  model_loader.py     config, adapter validation, lazy 4-bit + LoRA loading
  predictor.py        FactLLaMAPredictor and LlamaTextGenerator
  __init__.py         public exports
configs/factllama.json          repo-relative example configuration
requirements-factllama.txt      optional GPU inference dependencies
```

## Prompt format

The system instruction requires evidence-based verification, forbids outside
knowledge, and requests exactly one of `true`, `half`, `false`. The user
message contains:

```
Claim:
<claim>

Evidence:
1. <first evidence text>
2. <second evidence text>

Verdict:
```

Evidence can be plain strings or retrieval records with a `content`, `text`
or `evidence` field, so output from `src/retrieval` can be passed directly.
Text is preserved apart from whitespace trimming.

> The original fine-tuning notebook/template is not in this repository, so
> the wording above mirrors the documented format rather than an exact
> character-for-character copy. Compare it with the training notebook before
> drawing conclusions from local outputs.

## Setup

Python 3.10–3.12 with a CUDA GPU is recommended for 4-bit inference.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-factllama.txt
pip install pytest
```

The base checkpoint is gated: accept Meta's licence on Hugging Face, then
authenticate once.

```powershell
huggingface-cli login
# or set HF_TOKEN in the environment / .env
```

## Point the loader at the saved adapter

Copy the adapter folder from Google Drive:

```
My Drive/FactLLaMA Datasets/factllama_final_adapter_v2/
```

into the repository:

```
models/factllama/factllama_final_adapter_v2/
```

Expected essential files:

```
adapter_config.json
adapter_model.safetensors
tokenizer_config.json          # optional; base tokenizer is used if absent
tokenizer.json / special_tokens_map.json
```

Do **not** commit the adapter: `models/**` and weight/checkpoint patterns are
git-ignored. Alternatively, keep it outside the repo and point
`FACTLLAMA_ADAPTER_PATH` at the folder. No machine-specific path is hardcoded.

## Configuration

`configs/factllama.json` is the committed example. Relative `adapter_path`
values resolve against the repository root, so the loader works from any
working directory.

| Environment variable | Overrides |
| --- | --- |
| `FACTLLAMA_CONFIG` | Path to an alternative JSON config file |
| `FACTLLAMA_BASE_MODEL` | `base_model` |
| `FACTLLAMA_ADAPTER_PATH` | `adapter_path` |
| `FACTLLAMA_LOAD_IN_4BIT` | `load_in_4bit` |
| `FACTLLAMA_BNB_4BIT_QUANT_TYPE` | `bnb_4bit_quant_type` (`nf4`/`fp4`) |
| `FACTLLAMA_BNB_4BIT_COMPUTE_DTYPE` | `bnb_4bit_compute_dtype` |
| `FACTLLAMA_BNB_4BIT_USE_DOUBLE_QUANT` | `bnb_4bit_use_double_quant` |
| `FACTLLAMA_DEVICE_MAP` | `device_map` |
| `FACTLLAMA_TRUST_REMOTE_CODE` | `trust_remote_code` |
| `FACTLLAMA_MAX_NEW_TOKENS` | `max_new_tokens` |
| `FACTLLAMA_DO_SAMPLE` | `do_sample` |
| `FACTLLAMA_TEMPERATURE` | `temperature` |

Environment variables take precedence over the JSON file. Unknown keys and
invalid values raise `ConfigurationError` instead of being silently ignored.

Example override:

```powershell
$env:FACTLLAMA_ADAPTER_PATH = "D:\models\factllama_final_adapter_v2"
```

## Usage

```python
from src.factllama import FactLLaMAPredictor

predictor = FactLLaMAPredictor()  # nothing loads yet
result = predictor.predict(
    "The Eiffel Tower is located in Paris.",
    ["The Eiffel Tower is a landmark in Paris, France."],
)
print(result.verdict, result.raw_output)  # e.g. true Verdict: true
```

`verdict` is `None` when the model output contains no supported label; the
raw generation is always kept. The loaded model is cached for the lifetime of
the predictor/loader instance. Loading requires CUDA and `bitsandbytes` when
`load_in_4bit` is true.

## Tests

```powershell
python -m pytest tests/factllama -q
```

Tests use mocks/fakes and require no GPU, network, Hugging Face
authentication, model weights or adapter files.

## Errors

| Exception | Meaning |
| --- | --- |
| `AdapterNotFoundError` | Adapter folder or essential files missing |
| `MissingDependencyError` | `torch` / `transformers` / `peft` / `accelerate` / `bitsandbytes` not installed |
| `QuantizationUnsupportedError` | 4-bit requested without a usable CUDA GPU |
| `ModelAccessError` | Gated base model, missing HF token, upload/network failure |
| `ModelLoadError` | Other tokenizer/model/adapter loading failures |
| `PromptBuildError` | Empty claim or invalid/empty evidence |
| `VerdictParseError` / `None` verdict | Unparseable model output |

## Known limitations

- Adapter loading is implemented but **not yet verified** with the real
  `factllama_final_adapter_v2` files.
- 4-bit inference requires CUDA. `bitsandbytes` support on native Windows is
  limited; WSL2 or Linux is the safer path if CUDA setup fails.
- The API and frontend are intentionally out of scope for this change, and the
  retrieval module is unchanged.
