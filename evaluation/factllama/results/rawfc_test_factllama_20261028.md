# FactLLaMA RAWFC Test Evaluation

- Dataset: `C:\Users\Gagan\OneDrive\Desktop\MLProject\ML-major-project\data\RAWFC\test` (200 JSON files)
- Model: cached Llama-3.2-3B-Instruct + existing FactLLaMA LoRA adapter
- Generation: deterministic, existing predictor/prompt/parser, `max_new_tokens=16`

## Label mapping

| Original label | Canonical label |
|---|---|
| `true` | `true` |
| `correct attribution` | `true` |
| `Mostly True` | `true` |
| `mixture` | `half` |
| `unproven` | `half` |
| `false` | `false` |
| `misattributed` | `false` |
| `Mostly False` | `false` |

## Counts

| Count | Value |
|---|---:|
| total_files | 200 |
| successfully_evaluated | 200 |
| malformed | 0 |
| failed | 0 |
| invalid_or_unparsable_predictions | 0 |
| scored | 200 |

## Metrics

- Accuracy: **0.615000**
- Macro-F1: **0.607248**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| true | 0.591398 | 0.820896 | 0.687500 | 67 |
| half | 0.574074 | 0.462687 | 0.512397 | 67 |
| false | 0.698113 | 0.560606 | 0.621849 | 66 |

## Confusion matrix

Rows are actual labels; columns are predicted labels.

| Actual \ Predicted | true | half | false |
|---|---:|---:|---:|
| true | 55 | 6 | 6 |
| half | 26 | 31 | 10 |
| false | 12 | 17 | 37 |

## Invalid predictions and failures

- Invalid/unparsable predictions: **0**
- Prediction failures: **0**
- Malformed records: **0**
- Metrics include only successfully parsed predictions; invalid predictions and failures are not counted as correct.

## Baseline comparison

The previously reported baseline (61.5% accuracy, approximately 0.606 macro-F1) was not found with a reproducible RAWFC test-split evaluation script in the repository. It is treated as **not directly comparable** unless its exact split, label mapping, and scoring denominator are confirmed.

## Output

- JSON: `C:\Users\Gagan\OneDrive\Desktop\MLProject\ML-major-project\evaluation\factllama\results\rawfc_test_factllama_20261028.json`
- Markdown: `C:\Users\Gagan\OneDrive\Desktop\MLProject\ML-major-project\evaluation\factllama\results\rawfc_test_factllama_20261028.md`
