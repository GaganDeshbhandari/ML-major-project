# FactLLaMA inference

Evidence-based claim verification with `meta-llama/Llama-3.2-3B-Instruct`
and the QLoRA adapter `factllama_final_adapter_v2`.

> **Status:** the inference scaffolding and prompt/parsing/budget logic are
> implemented and unit-tested with mocks. The cached base model now loads
> successfully in the dedicated native-Windows CUDA environment, the saved
> adapter attaches successfully, and one deterministic synthetic smoke
> generation returned a valid verdict. The 61.5% accuracy / ~0.606 macro-F1
> figures come from the fine-tuning evaluation and are not re-measured here.

## Layout

```
src/factllama/
  prompt_builder.py   system/user prompt, evidence validation and budget
  output_parser.py    unambiguous true | half | false parsing
  model_loader.py     config, adapter validation, lazy 4-bit + LoRA loading
  predictor.py        FactLLaMAPredictor and LlamaTextGenerator
  __init__.py         public exports
configs/factllama.json        repo-relative example configuration
requirements-factllama.txt    direct inference pins (Windows / CUDA 13.0)
requirements-factllama.lock   fully pinned validated environment
requirements.txt              shared project dependencies (not inference-only)
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

Validated on 2026-10-27 with:

- Windows 11
- Python 3.12.10
- NVIDIA GeForce RTX 2050, 4 GB VRAM, driver 616.92
- PyTorch 2.14.1+cu130 (CUDA 13.0)
- Transformers 5.19.0, PEFT 0.21.2, Accelerate 1.15.0
- bitsandbytes 0.50.2

Keep this environment separate from the shared `venv312`; do not replace the
existing CPU-only environment.

```powershell
py -3.12 -m venv C:\Users\Gagan\venvs\factllama
C:\Users\Gagan\venvs\factllama\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-factllama.txt
```

If the target path already exists, inspect it before running the creation
command; do not overwrite an existing environment.

`requirements-factllama.txt` pins the direct inference dependencies and points
at the official CUDA 13.0 wheel index. `requirements-factllama.lock` records the
full transitive set from the validated environment. Do not install the shared
`requirements.txt` into this environment: it also contains retrieval and
prioritization dependencies that FactLLaMA does not need.

Validate the CUDA build before attempting inference:

```powershell
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

Validated output:

```text
torch: 2.14.1+cu130
CUDA build: 13.0
CUDA available: True
GPU: NVIDIA GeForce RTX 2050
```

The environment also passed a CUDA `bitsandbytes.nn.Linear4bit` smoke test on
`cuda:0` with BF16 input and finite output. This verifies the 4-bit kernel path,
not Llama model loading.

```powershell
@'
import torch
import bitsandbytes as bnb

layer = bnb.nn.Linear4bit(
    64, 32, bias=False, quant_type="nf4", compute_dtype=torch.bfloat16
).cuda()
x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)
with torch.no_grad():
    y = layer(x)
print(y.device, y.dtype, tuple(y.shape), bool(torch.isfinite(y).all()))
'@ | python -
```

Validated output:

```text
cuda:0 torch.bfloat16 (2, 32) True
```

A harmless `triton not found` warning may appear during imports; it only
disables PyTorch FLOP counting for Triton kernels and did not affect the smoke
test.

The base checkpoint is gated. Accept Meta's licence at
<https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct>, then authenticate:

```powershell
hf auth login
```

`huggingface-cli` is deprecated by `huggingface_hub` 1.x; use `hf auth login`.
The loader does not read `.env` files itself. If you use a token instead of the
CLI login, export `HF_TOKEN` in the process environment; never commit it.

## Point the loader at the saved adapter

The adapter is present locally at:

```
models/factllama/factllama_final_adapter_v2/
```

It was structurally validated on 2026-10-27:

- `adapter_config.json` and `adapter_model.safetensors` are present.
- The safetensors file is valid and contains 392 tensors: 28 layers x
  7 target modules x LoRA A/B.
- Rank is 16; alpha is 32; the base model is
  `meta-llama/Llama-3.2-3B-Instruct`.
- The tokenizer loads with Transformers 5.19.0 and exposes a chat template.
- `special_tokens_map.json` is not present; it is optional for this tokenizer.

Expected essential files:

```
adapter_config.json
adapter_model.safetensors
tokenizer_config.json          # optional; base tokenizer is used if absent
tokenizer.json                 # present in this adapter
chat_template.jinja            # present in this adapter
special_tokens_map.json        # optional; not required here
```

If the adapter is not present, copy it from
`My Drive/FactLLaMA Datasets/factllama_final_adapter_v2/` into
`models/factllama/factllama_final_adapter_v2/`. Do **not** commit it:
`models/**` and weight/checkpoint patterns are git-ignored. Alternatively, keep
it outside the repo and point `FACTLLAMA_ADAPTER_PATH` at the folder. No
machine-specific path is hardcoded.

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

Phase 2 validation ran this end-to-end usage path once with the cached base
snapshot and the saved adapter. It did not evaluate a dataset, benchmark
throughput, or start fine-tuning.

## Tests

```powershell
python -m pytest tests/factllama -q
```

Validated result in the dedicated environment on 2026-10-27:

```text
111 passed
```

Tests use mocks/fakes and require no GPU, network, Hugging Face
authentication, model weights or adapter files. They do **not** prove that the
real adapter or base model loads. Retrieval tests are not present on
`feature/factllama` (that code lives on `feature/retrieval`), so no retrieval
tests are run here.

## Phase 1 validation status

Validated in `C:\Users\Gagan\venvs\factllama` on 2026-10-27:

| Check | Result |
| --- | --- |
| Python | 3.12.10 |
| PyTorch / CUDA | 2.14.1+cu130, CUDA 13.0, `torch.cuda.is_available() == True` |
| GPU | NVIDIA GeForce RTX 2050 |
| bitsandbytes `Linear4bit` | Passed on `cuda:0`; BF16 output; all values finite |
| Adapter config and tokenizer | Loaded locally; PEFT 0.21.2 parsed the LoRA config |
| FactLLaMA mocked tests | 111 passed |
| `pip check` | No broken requirements |
| Base model access | Blocked: gated model, HTTP 401, no Hugging Face login |
| Adapter attachment and generation | Not run |

## Phase 2 real-model validation

Validated on 2026-10-28 with the cached snapshot at
`C:\Users\Gagan\.cache\huggingface\hub\models--meta-llama--Llama-3.2-3B-Instruct\snapshots\0cb88a4f764b7a12671c53f0838cd831a0843b95`:

| Check | Result |
| --- | --- |
| Cached model files | Present: config, tokenizer, and both safetensors shards |
| Architecture | `llama`, `LlamaForCausalLM`, 28 layers, hidden size 3072 |
| Quantized base loading | Passed with NF4, BF16 compute, and double quantization |
| Adapter attachment | Passed; `PeftModelForCausalLM`, LoRA, active adapter `default` |
| Device placement | `cuda:0`; no CPU offload observed; `hf_device_map` was `None` |
| GPU memory | 2,340,057,088 bytes allocated / 2,487,222,272 bytes reserved after load |
| Synthetic deterministic generation | Passed; raw output `true`, parsed verdict `true` |
| FactLLaMA regression tests | `111 passed` |
| Dependency check | `No broken requirements found` |

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

- The real-model validation used one synthetic claim/evidence pair only. Do
  not treat it as an evaluation or benchmark.
- The validated environment uses native Windows and CUDA 13.0. The CUDA build
  and bitsandbytes kernel path passed a small smoke test, but a 3B model in
  4-bit may still exceed the RTX 2050's 4 GB VRAM once embeddings, KV cache,
  activations, and display overhead are included. `device_map="auto"` may
  offload layers to CPU; check the loaded model's device map before evaluating
  throughput.
- WSL2 is a fallback, not the validated path. It is not installed on this
  machine, so using it would require a separate setup and reboot.
- The evidence budget uses a character-based approximation, not exact
  tokenizer counts.
- The API, frontend and retrieval integration are intentionally out of scope;
  the retrieval module is unchanged.
