# NLI Stance Model Comparison

Generated: 2026-10-26T05:53:24.840322+00:00

## Scope

This comparison uses the frozen diagnostic dataset and the existing benchmark metric definitions. It compares measured results only; it does not establish general model quality.

## Dataset

| Property | Value |
| --- | --- |
| SHA-256 | 198b783ab6d457515bde8924e3c9f4ea1f227f2a686fcdb42fc19dd285b32178 |
| Total examples | 90 |
| Class counts | {"not_enough_info": 30, "refutes": 30, "supports": 30} |

## Models

| Label | Model | Configured revision | Device requested | Resolved device | Dtype | Sequence length | Configured batch size | Confidence threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Existing xsmall | cross-encoder/nli-deberta-v3-xsmall | a150876415327c80daeff35ca6f68f5ed8cf5c24 | auto | cpu | torch.float32 | 512 | 2 | None |
| Alternative base | MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 6f5cf0a2b59cabb106aca4c287eed12e357e90eb | auto | cpu | torch.float16 | 512 | 2 | None |

## Run counts

| Model | Total examples | Evaluated | Inference failures | Coverage |
| --- | --- | --- | --- | --- |
| Existing xsmall | 90 | 90 | 0 | 1.0000 |
| Alternative base | 90 | 90 | 0 | 1.0000 |

## Headline comparison

| Metric | Existing xsmall | Alternative base | Delta (alt - baseline) |
| --- | --- | --- | --- |
| Accuracy | 0.8667 | 0.8778 | 0.0111 |
| Macro-F1 | 0.8647 | 0.8759 | 0.0112 |
| SUPPORTS recall | 0.9667 | 0.9000 | -0.0667 |
| REFUTES recall | 0.9000 | 0.9667 | 0.0667 |
| NOT_ENOUGH_INFO recall | 0.7333 | 0.7667 | 0.0333 |
| Inference failures | 0 | 0 | 0 |
| Coverage | 1.0000 | 1.0000 | 0.0000 |
| Evaluated examples | 90 | 90 | 0 |
| Average warm inference seconds/example | 0.173 | 2.526 | 2.353 |
| Initial loading seconds | 37.842 | 39.754 | 1.912 |
| Total attempt seconds | 53.241 | 264.603 | 211.362 |
| Peak working set MB | 871.3 | 871.8 | 0.5 |

## Per-class metrics

### Existing xsmall

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| supports | 0.8056 | 0.9667 | 0.8788 | 30 |
| refutes | 0.8710 | 0.9000 | 0.8852 | 30 |
| not_enough_info | 0.9565 | 0.7333 | 0.8302 | 30 |

### Alternative base

| Label | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| supports | 0.8710 | 0.9000 | 0.8852 | 30 |
| refutes | 0.9062 | 0.9667 | 0.9355 | 30 |
| not_enough_info | 0.8519 | 0.7667 | 0.8070 | 30 |

## Confusion matrices

### Existing xsmall

Rows are gold labels; columns are predicted labels.

| Gold \ Predicted | supports | refutes | not_enough_info |
| --- | --- | --- | --- |
| supports | 29 | 0 | 1 |
| refutes | 3 | 27 | 0 |
| not_enough_info | 4 | 4 | 22 |

| Gold | Predicted | Count |
| --- | --- | --- |
| not_enough_info | supports | 4 |
| not_enough_info | refutes | 4 |
| refutes | supports | 3 |
| supports | not_enough_info | 1 |

### Alternative base

Rows are gold labels; columns are predicted labels.

| Gold \ Predicted | supports | refutes | not_enough_info |
| --- | --- | --- | --- |
| supports | 27 | 0 | 3 |
| refutes | 0 | 29 | 1 |
| not_enough_info | 4 | 3 | 23 |

| Gold | Predicted | Count |
| --- | --- | --- |
| not_enough_info | supports | 4 |
| supports | not_enough_info | 3 |
| not_enough_info | refutes | 3 |
| refutes | not_enough_info | 1 |

## Confusion count comparison

| Confusion | Existing xsmall | Alternative base |
| --- | --- | --- |
| NOT_ENOUGH_INFO -> SUPPORTS | 4 | 4 |
| NOT_ENOUGH_INFO -> REFUTES | 4 | 3 |
| REFUTES -> SUPPORTS | 3 | 0 |
| SUPPORTS -> NOT_ENOUGH_INFO | 1 | 3 |

## Error transitions

| Item | Value |
| --- | --- |
| Baseline misclassified | 12 |
| Alternative misclassified | 11 |
| Baseline high-confidence errors (>= 0.9) | 7 |
| Alternative high-confidence errors (>= 0.9) | 6 |

- Common errors: `NEI-002`, `NEI-007`, `NEI-009`, `NEI-013`, `NEI-024`, `REF-022`, `SUP-010`
- Fixed by alternative: `NEI-016`, `NEI-017`, `NEI-021`, `REF-006`, `REF-019`
- New in alternative: `NEI-006`, `NEI-025`, `SUP-014`, `SUP-026`

## Misclassified examples

The tables preserve example IDs. The selected bullets emphasize NOT_ENOUGH_INFO-to-stance errors, REFUTES-to-SUPPORTS errors, and incorrect predictions with confidence >= 0.9.

### Existing xsmall

| ID | Gold | Predicted | Confidence | Category |
| --- | --- | --- | --- | --- |
| SUP-010 | supports | not_enough_info | 0.9846 | direct_support |
| REF-006 | refutes | supports | 0.6965 | direct_contradiction |
| REF-019 | refutes | supports | 0.4980 | date |
| REF-022 | refutes | supports | 0.9634 | numerical |
| NEI-002 | not_enough_info | refutes | 0.9996 | related_non_decisive |
| NEI-007 | not_enough_info | supports | 0.9784 | overlapping_vocabulary |
| NEI-009 | not_enough_info | supports | 0.9321 | related_non_decisive |
| NEI-013 | not_enough_info | refutes | 0.4016 | related_non_decisive |
| NEI-016 | not_enough_info | supports | 0.7975 | related_non_decisive |
| NEI-017 | not_enough_info | supports | 0.8018 | related_non_decisive |
| NEI-021 | not_enough_info | refutes | 0.9766 | overlapping_vocabulary |
| NEI-024 | not_enough_info | refutes | 0.9944 | related_non_decisive |

Selected high-priority or high-confidence errors:

- `SUP-010` gold=`supports` predicted=`not_enough_info` confidence=0.9846
  - Claim: The Pacific Ocean is the largest ocean on Earth.
  - Evidence: The Pacific covers more area than any other ocean.
- `REF-006` gold=`refutes` predicted=`supports` confidence=0.6965
  - Claim: Earth is the center of the Solar System.
  - Evidence: Earth orbits the Sun, which is at the center of the Solar System.
- `REF-019` gold=`refutes` predicted=`supports` confidence=0.4980
  - Claim: Christopher Columbus reached the Americas in 1493.
  - Evidence: Columbus's first voyage to the Americas arrived in 1492.
- `REF-022` gold=`refutes` predicted=`supports` confidence=0.9634
  - Claim: A common year has 366 days.
  - Evidence: A non-leap year contains 365 days.
- `NEI-002` gold=`not_enough_info` predicted=`refutes` confidence=0.9996
  - Claim: Pure water freezes at 0 degrees Celsius at standard pressure.
  - Evidence: Pure water boils at 100 degrees Celsius at standard pressure.
- `NEI-007` gold=`not_enough_info` predicted=`supports` confidence=0.9784
  - Claim: Two plus two equals four.
  - Evidence: Two multiplied by two equals four.
- `NEI-009` gold=`not_enough_info` predicted=`supports` confidence=0.9321
  - Claim: Tokyo is in Japan.
  - Evidence: Tokyo is one of the world's most populous metropolitan areas.
- `NEI-013` gold=`not_enough_info` predicted=`refutes` confidence=0.4016
  - Claim: One hour contains 60 minutes.
  - Evidence: One day contains 24 hours.
- `NEI-016` gold=`not_enough_info` predicted=`supports` confidence=0.7975
  - Claim: Mammals are warm-blooded animals.
  - Evidence: Mammals nurse their young with milk.
- `NEI-017` gold=`not_enough_info` predicted=`supports` confidence=0.8018
  - Claim: Berlin is the capital of Germany.
  - Evidence: Berlin is the largest city in Germany by population.
- `NEI-021` gold=`not_enough_info` predicted=`refutes` confidence=0.9766
  - Claim: A common year has 365 days.
  - Evidence: A leap year has 366 days.
- `NEI-024` gold=`not_enough_info` predicted=`refutes` confidence=0.9944
  - Claim: Ice is the solid form of water.
  - Evidence: Ice floats on liquid water.

### Alternative base

| ID | Gold | Predicted | Confidence | Category |
| --- | --- | --- | --- | --- |
| SUP-010 | supports | not_enough_info | 0.9741 | direct_support |
| SUP-014 | supports | not_enough_info | 0.9684 | direct_support |
| SUP-026 | supports | not_enough_info | 0.9654 | numerical |
| REF-022 | refutes | not_enough_info | 0.5079 | numerical |
| NEI-002 | not_enough_info | refutes | 0.9972 | related_non_decisive |
| NEI-006 | not_enough_info | supports | 0.8941 | related_non_decisive |
| NEI-007 | not_enough_info | supports | 0.9864 | overlapping_vocabulary |
| NEI-009 | not_enough_info | supports | 0.7835 | related_non_decisive |
| NEI-013 | not_enough_info | refutes | 0.6104 | related_non_decisive |
| NEI-024 | not_enough_info | refutes | 0.9774 | related_non_decisive |
| NEI-025 | not_enough_info | supports | 0.6725 | related_non_decisive |

Selected high-priority or high-confidence errors:

- `SUP-010` gold=`supports` predicted=`not_enough_info` confidence=0.9741
  - Claim: The Pacific Ocean is the largest ocean on Earth.
  - Evidence: The Pacific covers more area than any other ocean.
- `SUP-014` gold=`supports` predicted=`not_enough_info` confidence=0.9684
  - Claim: Antarctica is the coldest continent.
  - Evidence: Antarctica has lower average temperatures than any other continent.
- `SUP-026` gold=`supports` predicted=`not_enough_info` confidence=0.9654
  - Claim: Jupiter is the largest planet in the Solar System.
  - Evidence: Jupiter has a greater mass and volume than any other planet orbiting the Sun.
- `NEI-002` gold=`not_enough_info` predicted=`refutes` confidence=0.9972
  - Claim: Pure water freezes at 0 degrees Celsius at standard pressure.
  - Evidence: Pure water boils at 100 degrees Celsius at standard pressure.
- `NEI-006` gold=`not_enough_info` predicted=`supports` confidence=0.8941
  - Claim: The human heart pumps blood.
  - Evidence: The human heart is about the size of a fist.
- `NEI-007` gold=`not_enough_info` predicted=`supports` confidence=0.9864
  - Claim: Two plus two equals four.
  - Evidence: Two multiplied by two equals four.
- `NEI-009` gold=`not_enough_info` predicted=`supports` confidence=0.7835
  - Claim: Tokyo is in Japan.
  - Evidence: Tokyo is one of the world's most populous metropolitan areas.
- `NEI-013` gold=`not_enough_info` predicted=`refutes` confidence=0.6104
  - Claim: One hour contains 60 minutes.
  - Evidence: One day contains 24 hours.
- `NEI-024` gold=`not_enough_info` predicted=`refutes` confidence=0.9774
  - Claim: Ice is the solid form of water.
  - Evidence: Ice floats on liquid water.
- `NEI-025` gold=`not_enough_info` predicted=`supports` confidence=0.6725
  - Claim: Abraham Lincoln was a U.S. president.
  - Evidence: Abraham Lincoln was assassinated in 1865.

## Reading the comparison

- Rule-based headline verdict: `alternative`.
- Assessment: Marginal headline improvement; the gain is not concentrated in NOT_ENOUGH_INFO recall, and warm inference is more than 5x slower. On this evidence, the alternative does not justify replacing the production default.
- Headline: the alternative is higher on accuracy and macro-F1. For the stated NOT_ENOUGH_INFO concern, the baseline has higher NOT_ENOUGH_INFO precision (`0.9565` vs `0.8519`) while the alternative has slightly higher recall (`0.7333` vs `0.7667`); the alternative is not a clear win on that concern.
- Accuracy delta (alternative - baseline): `0.0111` (+1 example).
- Macro-F1 delta: `0.0112`.
- NOT_ENOUGH_INFO recall delta: `0.0333`; NOT_ENOUGH_INFO F1 delta: `-0.0232`; precision `0.9565` -> `0.8519`.
- Improvement concentrated in NOT_ENOUGH_INFO recall: `no`.
- Per-class F1 deltas: SUPPORTS `0.0065`, REFUTES `0.0502`, NOT_ENOUGH_INFO `-0.0232`.
- Confusion counts: NOT_ENOUGH_INFO -> SUPPORTS `4` -> `4`; NOT_ENOUGH_INFO -> REFUTES `4` -> `3`; REFUTES -> SUPPORTS `3` -> `0`; SUPPORTS -> NOT_ENOUGH_INFO `1` -> `3`.
- Warm runtime ratio (alternative / baseline): `14.6010`; total runtime ratio: `4.9699`.
- Peak working-set delta MB: `0.5`.
- Misclassified examples: baseline `12`, alternative `11`; high-confidence errors (>= 0.9): baseline `7`, alternative `6`.

The verdict is a simple rule: `alternative` means both accuracy and macro-F1 are higher; `baseline` means both are lower; `mixed` means the two headline metrics disagree. The assessment is a rule-based interpretation, not a statistical test.

## Measurement notes and caveats

- All classification metrics and timings are read from the benchmark JSON reports; they are not recomputed here.
- Peak memory is the process peak working set measured with Windows GetProcessMemoryInfo. It includes the Python process, imported libraries, tokenizer, model, and inference workspace; it is not isolated model-weight memory.
- The dataset is the frozen 90-example diagnostic set. Results are not a representative benchmark or proof of general reliability.
- With 90 examples, one example changes accuracy by about 1.11 percentage points; small differences are not statistically robust.
- No confidence threshold was configured for either model.
- Model dtypes differ: baseline=torch.float32, alternative=torch.float16. The existing classifier loaded each checkpoint's configured dtype; CPU runtime comparisons may be affected by dtype-specific kernels.
- The baseline metrics reproduce the prior nli_stance_benchmark.json exactly; the fresh baseline run was used to add process peak-memory and dtype metadata under the same runner.
- Supplementary model-weight sizes from safetensors metadata (parameters only, not peak memory): xsmall 270.2 MiB (F32), base 351.8 MiB (F16); the base checkpoint weights are about 81.6 MiB larger.

## Provenance

| Report | Path |
| --- | --- |
| Existing xsmall | evaluation\evidence_conflict\results\nli_stance_benchmark_xsmall_remeasured.json |
| Alternative base | evaluation\evidence_conflict\results\nli_stance_benchmark_deberta_v3_base.json |
