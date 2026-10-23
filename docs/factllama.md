# FactLLaMA inference

Evidence-based claim verification with `meta-llama/Llama-3.2-3B-Instruct`
and the QLoRA adapter `factllama_final_adapter_v2`.

> **Status:** the inference scaffolding and prompt/parsing/budget logic are
> implemented and unit-tested with mocks. The saved adapter has **not** been
> loaded or run yet, so real end-to-end inference is unverified. The 61.5%
> accuracy / ~0.606 macro-F1 figures come from the fine-tuning evaluation and
> are not re-measured here.

## Layout

```
src/factllama/
  prompt_builder.py   system/user prompt, evidence validation and budget
  output_parser.py    unambiguous true | half | false parsing
  model_loader.py     config, adapter validation, lazy 4-bit + LoRA loading
  predictor.py        FactLLaMAPredictor and LlamaTextGenerator
  __init__.py         public exports
configs/factllama.json   repo-relative example configuration
requirements.txt         inference dependencies (QLoRA / 4-bit)
```

## Prompt format

The system prompt is the exact text from cells 8 and 10 of the fine-tuning
notebook:

```
You are a fact-checking assistant. Evaluate the claim using only the provided evidence. Output exactly one verdict: true, half, or false.
```

The user message has this exact structure:

```
Claim: <claim>

Evidence:
[1] <first evidence text>
[2] <second evidence text>

Verdict:
```

- The claim appears on the same line as `Claim:`; only surrounding whitespace
  is trimmed. It is never truncated by the evidence budget.
- Evidence is numbered `[1]`, `[2]`, ... in the original order.
- Evidence can be plain strings or records with a `content`, `text` or
  `evidence` field. Only that text is sent to the model; titles, URLs and
  other source metadata are ignored. No retrieval code is imported.
- The tokenizer's chat template is applied afterwards by
  `LlamaTextGenerator`. That wrapper has not been compared against the
  fine-tuning notebook, so only the system/user text is confirmed.

## Verdict parsing

`parse_verdict()` accepts only `true`, `half` and `false`.

- Capitalisation, surrounding whitespace and markdown code fences are
  tolerated.
- A `Verdict:` prefix is supported. When present, only the text after it is
  considered, so a label mentioned in an explanation before the marker is
  ignored.
- Labels must be whole words: `truthful` and `falsehood` do not match.
- **Conflicting labels are rejected** instead of taking the first one. For
  example, `Verdict: true or false`, `true and false`, or an explanation that
  mentions two different labels raises `VerdictParseError`.
- Multiple `Verdict:` markers, an empty output, and a marker with no label are
  also rejected.
- `try_parse_verdict()` returns `None` for any rejected output;
  `FactLLaMAPredictor` keeps the raw generation in `Prediction.raw_output`.

## Evidence budget

Evidence is capped by an approximate token budget before the prompt is built.
The defaults are:

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_evidence_tokens` | `1024` | Total estimated evidence tokens |
| `evidence_chars_per_token` | `4.0` | Characters per token estimate |
| min truncation size | `32` tokens (capped by the budget) | Internal; avoids tiny fragments |

Behaviour (`select_evidence`):

1. Passages are kept in their original order.
2. A passage is included whole when it fits the remaining budget.
3. The first passage that does not fit is truncated to the remaining budget
   (preferring a word/line boundary) when at least the minimum remains
   (`32` by default, capped by `max_evidence_tokens`). All later passages are
   then excluded.
4. If fewer than that minimum remain, the passage and all later passages are
   excluded.
5. The claim is outside this budget and is never truncated.

The budget is **approximate**: it is computed from characters, not from the
model tokenizer. It covers evidence text only; the system prompt, claim and
generation tokens are accounted for separately by the model's context window.
`build_prompt()` returns an `EvidenceSelection` with `estimated_tokens`,
`truncated` and `excluded`; predictions expose the same information as
`evidence_used` (estimated evidence tokens), `evidence_truncated` and
`evidence_excluded`.
`build_user_prompt()` remains available when only the rendered string is
needed.

## Setup

Python 3.10–3.12 with a CUDA GPU is recommended for 4-bit inference.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
```

`requirements.txt` contains the inference dependencies (`torch`,
`transformers`, `peft`, `accelerate`, `bitsandbytes`). Install the CUDA build
of PyTorch that matches your driver if the default wheel is CPU-only. Unit
tests only need `pytest`.

The base checkpoint is gated: accept Meta's licence on Hugging Face, then
authenticate once.

```powershell
huggingface-cli login
# or set HF_TOKEN in the environment
```

The loader does not read `.env` files itself; export `HF_TOKEN` in the process
environment if you authenticate with a token instead of `huggingface-cli login`.

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
working directory. Environment variables take precedence over the JSON file;
unknown keys and invalid values raise `ConfigurationError`.

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
| `FACTLLAMA_MAX_EVIDENCE_TOKENS` | `max_evidence_tokens` |
| `FACTLLAMA_EVIDENCE_CHARS_PER_TOKEN` | `evidence_chars_per_token` |

Example override:

```powershell
$env:FACTLLAMA_ADAPTER_PATH = "<absolute-path-to-factllama_final_adapter_v2>"
$env:FACTLLAMA_MAX_EVIDENCE_TOKENS = "2048"
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
print(result.evidence_truncated, result.evidence_excluded)
```

`verdict` is `None` when the model output is ambiguous or unparseable; the raw
generation is always kept. The loaded model is cached for the lifetime of the
predictor/loader instance. Loading requires CUDA and `bitsandbytes` when
`load_in_4bit` is true, and the checks run before any model weights are loaded.
Importing the package never imports PyTorch or downloads weights.

## Tests

```powershell
python -m pytest tests/factllama -q
```

Tests use mocks/fakes and require no GPU, network, Hugging Face
authentication, model weights or adapter files. Retrieval tests are not
present on `feature/factllama` (that code lives on `feature/retrieval`), so no
retrieval tests are run here.

## Errors

| Exception | Meaning |
| --- | --- |
| `AdapterNotFoundError` | Adapter folder or essential files missing |
| `MissingDependencyError` | `torch` / `transformers` / `peft` / `accelerate` / `bitsandbytes` not installed |
| `QuantizationUnsupportedError` | 4-bit requested without a usable CUDA GPU |
| `ModelAccessError` | Gated base model, missing HF token, upload/network failure |
| `ModelLoadError` | Other tokenizer/model/adapter loading failures |
| `PromptBuildError` | Empty claim, invalid/empty evidence or invalid budget |
| `VerdictParseError` / `None` verdict | Unparseable or conflicting model output |

## Known limitations

- Adapter loading is implemented but **not yet verified** with the real
  `factllama_final_adapter_v2` files. Do not treat mocked tests as evidence
  that inference works.
- The local development environment used for this work is CPU-only and lacks
  `peft`/`bitsandbytes`, so 4-bit loading could not be exercised.
- 4-bit inference requires CUDA. `bitsandbytes` support on native Windows is
  limited; WSL2 or Linux is the safer path if CUDA setup fails.
- The evidence budget uses a character-based approximation, not exact
  tokenizer counts.
- The API, frontend and retrieval integration are intentionally out of scope;
  the retrieval module is unchanged.
