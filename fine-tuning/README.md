# Fine-Tuning — Text-to-SQL (AI League PS5)

**Task:** Natural language question → executable SQL query  
**Model:** `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` + LoRA  
**Platform:** MacBook M4 Air, 24GB Unified Memory — Apple MLX (no CUDA)  
**Submission notebook:** `notebooks/local_mac_mlx.ipynb`

---

## Mandatory Steps — Completion Map

| # | Step | Notebook Section | Status |
|---|------|-----------------|--------|
| 1 | Dataset cleaning and build | §1 — cells 5–8 | ✅ |
| 2 | Model choice and baseline benchmark | §2 — cells 10–12 | ✅ |
| 3 | Training strategy | §3 — markdown cell | ✅ |
| 4 | Hyperparameter justification | §4 — markdown + training cell | ✅ |
| 5 | Pre vs post output comparison | §5 — cells 20–21 | ✅ |
| 6 | Evaluation + loss curves | §6 — cells 23–26 | ✅ |
| 7 | Production thinking | §7 — cells 28 | ✅ |
| + | Bonus: Safety guardrails | Bonus — cell 30 | ✅ |

---

## Final Results

| Metric | Base Model | Fine-tuned | Delta |
|---|---|---|---|
| **Exact Match (EM)** | 3.00% | **69.87%** | +66.87pp (23× improvement) |
| **Exec Accuracy (EX)** | 88.66% | **94.08%** | +5.42pp |

**Complexity breakdown (fine-tuned, 1,500 examples):**

| Complexity | EM | EX | n |
|---|---|---|---|
| easy | 72.20% | 94.86% | 1,439 |
| medium | 18.18% | 76.19% | 22 |
| hard | 29.41% | 72.73% | 17 |
| extra_hard | 0.00% | 68.42% | 22 |

---

## Project Structure

```
fine-tuning/
├── README.md                         ← you are here
├── REPORT.md                         ← full written report (all 7 steps + bonus)
├── requirements.txt                  ← pip dependencies
├── configs/
│   └── qlora.yaml                    ← QLoRA hyperparameter config (Kaggle reference)
├── src/
│   ├── data_pipeline.py              ← load / clean / format / split datasets
│   ├── evaluate.py                   ← EM + execution accuracy + error taxonomy
│   └── inference.py                  ← generate_sql(), batch_generate_sql(), FastAPI stub
└── notebooks/
    ├── local_mac_mlx.ipynb           ← PRIMARY submission notebook (all 7 steps, real outputs)
    ├── kaggle_master.ipynb           ← Kaggle reference notebook (7B model, archived)
    ├── mlx_data/
    │   ├── train.jsonl               ← 2,000 training examples
    │   ├── valid.jsonl               ← ~6,200 validation examples
    │   └── test.jsonl                ← ~3,900 held-out test examples
    └── mlx_output/
        ├── best_adapter/             ← LoRA adapter at iter 200 (best val loss 0.763)
        ├── adapters/                 ← all checkpoints
        ├── serve.py                  ← FastAPI production stub
        ├── loss_curves.png           ← train/val loss + zoomed convergence plot
        ├── eda.png                   ← dataset complexity distribution
        ├── evaluation_charts.png     ← Base vs FT bar chart + error taxonomy pie
        ├── baseline_predictions.csv  ← base model predictions (100 examples)
        └── pre_post_comparison.csv   ← side-by-side examples (8, all complexity levels)
```

---

## Dataset

**Source:** `b-mc2/sql-create-context` (HuggingFace) — 78,577 NL→SQL pairs  
Each row: `question` + `context` (CREATE TABLE DDL) + `answer` (gold SQL)

**Why this dataset:** Schema is embedded inline in every row — enables execution accuracy
via in-memory SQLite without external database files.

**Cleaning pipeline:**

| Step | What removed | Why |
|---|---|---|
| Null/empty rows | Missing question, context, or SQL | Can't train on incomplete examples |
| SQL parse validation | Unparseable strings (via sqlparse) | Wrong training signal |
| Non-SELECT filter | INSERT/UPDATE/DELETE/DROP/DDL | Task is read-only |
| Deduplication | Exact question duplicates (MD5) | Prevents train/test leakage |
| Complexity labeling | — | Stratified split + per-bucket evaluation |

**Split (stratified by complexity):**

| Split | Size | Purpose |
|---|---|---|
| Train | 2,000 (sampled) | Fine-tuning |
| Validation | ~6,200 | Early stopping |
| Test (held-out) | ~3,900 | Final evaluation only |

---

## Model

**`mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`**

| Criterion | Justification |
|---|---|
| Size (3B) | Fits in 5.1GB of 24GB unified memory. Trains in ~40 min vs 10+ hrs for 7B. |
| Coder family | Pre-trained on SQL-heavy code corpus. Baseline EX already 88.66%. |
| 4-bit MLX | Apple-native weights. No CUDA/bitsandbytes. Runs on Metal GPU. |
| Apache 2.0 | Production-usable without restriction. |

---

## Training

**Method:** Supervised Fine-Tuning with LoRA via `mlx_lm lora`

```bash
python -m mlx_lm lora \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --train \
  --data notebooks/mlx_data \
  --num-layers 16 \
  --batch-size 4 \
  --iters 1500 \
  --val-batches 25 \
  --learning-rate 1e-4 \
  --steps-per-report 10 \
  --steps-per-eval 100 \
  --save-every 100 \
  --adapter-path notebooks/mlx_output/adapters \
  --max-seq-length 512
```

**Early stopped at iter 200** — val loss 0.763 (minimum). Rose to 0.832 at iter 300.

---

## Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| `lora_rank` | 16 | 6.65M trainable params (0.216% of 3B) — sufficient for format alignment |
| `num_layers` | 16 | Last 16 transformer layers — most task-specific representations |
| `learning_rate` | 1e-4 | MLX LoRA standard (full-precision adapter weights, not quantized gradients) |
| `iters` | 200 (early stopped) | Best val loss at iter 200; overfitting onset at iter 300 |
| `batch_size` | 4 | Peak memory 5.1GB — leaves 19GB free on M4 Air |
| `max_seq_length` | 512 | Covers 95th percentile of prompt+SQL lengths |

---

## Prompt Format

```
### Task
Generate a SQL query to answer the following question.

### Database Schema
CREATE TABLE employees (id INT, name TEXT, salary FLOAT, dept_id INT);

### Question
What is the average salary?

### SQL
SELECT AVG(salary) FROM employees
```

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Exact Match (EM)** | Normalized string equality after keyword casing + whitespace collapse |
| **Execution Accuracy (EX)** | Run gold + predicted SQL on in-memory SQLite, compare result sets |
| **Complexity Breakdown** | EM + EX per bucket: easy / medium / hard / extra_hard |
| **Error Taxonomy** | Classify failures: wrong_table / syntax_error / schema_skipped / wrong_column |

---

## Production Serving

```
POST /generate  {question: str, schema: str}  →  {sql: str}
```

- FastAPI stub: `mlx_output/serve.py`
- Run: `uvicorn serve:app --host 0.0.0.0 --port 8000`
- Adapter loads with base model at startup (~5.1GB total)
- Greedy decoding (temperature=0) for deterministic output
- Non-SELECT output blocked before returning

**Hosting options:**

| Option | Latency | Cost |
|---|---|---|
| Local Mac (MLX) | ~0.5–1s | $0 |
| HuggingFace Spaces | ~2–5s | $0 |
| Modal serverless | ~1s | Per-request |

---

## Safety Guardrails (Bonus)

Three layers:
1. **Post-generation validation** — reject non-SELECT, block destructive keywords (DELETE/DROP/etc.)
2. **PII guard** — block queries referencing password/ssn/credit_card fields not present in the schema
3. **Schema grounding** — schema is injected at request time; model never has cross-database access

All 3 guardrail tests pass (see notebook Bonus cell).
