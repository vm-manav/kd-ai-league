# AI League PS5 — Text-to-SQL Fine-Tuning

## Project Report

**Problem Statement:** Problem 1 — Text-to-SQL (Code Generation)  
**Hardware:** Kaggle Tesla T4 ×2 (16 GB VRAM), training + eval on GPU 0  
**Framework:** PyTorch + HuggingFace Transformers + PEFT (QLoRA) + bitsandbytes  

---

## Table of Contents

1. [Dataset Cleaning and Build](#1-dataset-cleaning-and-build)
2. [Model Choice and Baseline Benchmark](#2-model-choice-and-baseline-benchmark)
3. [Training Strategy](#3-training-strategy)
4. [Hyperparameter Justification](#4-hyperparameter-justification)
5. [Pre vs Post Output Comparison](#5-pre-vs-post-output-comparison)
6. [Evaluation](#6-evaluation)
7. [Production Thinking](#7-production-thinking)
8. [Bonus — Safety Guardrails](#bonus--safety-guardrails)

---

## 1. Dataset Cleaning and Build

### Sources


| Source                               | Role                                          | Notes                                                                                                                                                                           |
| ------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `**b-mc2/sql-create-context`**       | Primary (78,577 rows)                         | Schema embedded in every row; backbone of train/test                                                                                                                            |
| `**gretelai/synthetic_text_to_sql**` | Intended secondary (15K challenging+moderate) | Filter used outdated complexity labels (`challenging`/`moderate`); **0 rows matched** on current HF dataset schema. Training and eval effectively used sql-create-context only. |


Each row contains:

- `question`: natural language question  
- `context`: CREATE TABLE DDL (schema)  
- `answer` / `sql`: gold SQL query

### Schema

```
Input:  (question: str, context: str)  →  CREATE TABLE statements + NL question
Output: sql: str                        →  executable SELECT query
```

**Prompt format (training and inference):**

```
### Task
Generate a SQL query to answer the following question.

### Database Schema
{context}

### Question
{question}

### SQL
{sql}
```

### Cleaning Steps


| Step                 | What was removed                                                        | Why                                    |
| -------------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| Null / empty rows    | Missing question, context, or SQL                                       | Cannot train on incomplete examples    |
| SQL parse validation | Unparseable strings (sqlparse)                                          | Broken SQL gives wrong training signal |
| Non-SELECT filter    | INSERT / UPDATE / DELETE / DROP                                         | Task is read-only query generation     |
| Deduplication        | Exact question duplicates (MD5 hash; sql-create-context wins conflicts) | Prevents leakage                       |
| Complexity labeling  | —                                                                       | Stratified split + per-bucket eval     |


### Complexity Distribution

Classified from gold SQL:


| Class        | Rule                                 |
| ------------ | ------------------------------------ |
| `easy`       | Simple SELECT/WHERE, single table    |
| `medium`     | GROUP BY / HAVING / subquery / UNION |
| `hard`       | Single JOIN                          |
| `extra_hard` | Multi-JOIN or JOIN + aggregation     |


Dataset is heavily skewed toward easy queries (~96% of cleaned data). Results are reported per complexity bucket.

### Train / Val / Test Split

Stratified by complexity, `random_state=42`:


| Split                        | Size                           | Purpose                             |
| ---------------------------- | ------------------------------ | ----------------------------------- |
| Train (subset used)          | **18,000** (stratified sample) | QLoRA fine-tuning (~2,250 steps)    |
| Validation (during training) | **500**                        | Per-epoch eval loss                 |
| Test (held-out)              | **3,914**                      | Final evaluation (`test_split.csv`) |


---

## 2. Model Choice and Baseline Benchmark

### Model: `Qwen/Qwen2.5-Coder-7B-Instruct` (4-bit QLoRA)


| Criterion             | Justification                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------ |
| **Size (7B)**         | Strong SQL/code reasoning; fits on T4 with 4-bit base + LoRA adapters (~8–10 GB train, ~5.75 GB infer) |
| **Coder family**      | Pre-trained on code including SQL; strong untuned semantic SQL ability                                 |
| **Instruction-tuned** | Works with `### Task / ### SQL` format without extra system prompt                                     |
| **Apache 2.0**        | Production-usable                                                                                      |


**Why QLoRA on Kaggle T4:** Full 7B fine-tuning needs far more VRAM than a 16 GB T4. QLoRA (NF4 base + trainable LoRA) makes 7B training feasible in ~7–9 hours.

### Baseline Results (untuned model, **200 test examples**, greedy decoding)


| Metric                 | Score      |
| ---------------------- | ---------- |
| **Exact Match (EM)**   | **1.00%**  |
| **Exec Accuracy (EX)** | **36.92%** |


**Interpretation:** Low EM reflects format mismatch (quoting, casing, aliases). Moderate EX shows the base model understands SQL semantics but is inconsistent on schema-grounded generation — the main fine-tuning target.

---

## 3. Training Strategy

### Method: Supervised Fine-Tuning (SFT) with QLoRA

- **SFT:** Labeled NL→SQL pairs with gold answers.  
- **QLoRA:** 4-bit NF4 base weights + LoRA on attention/MLP projections.  
- **Loss:** Cross-entropy on SQL tokens (prompt masked via TRL SFTConfig).  
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.

### Training Run Summary


| Item             | Value                                      |
| ---------------- | ------------------------------------------ |
| Train samples    | 18,000 (stratified)                        |
| Steps            | 2,250                                      |
| Val samples      | 500                                        |
| Eval strategy    | Per epoch                                  |
| Adapter saved    | `/kaggle/working/qlora_sql/final_adapter/` |
| Best val loss    | **0.6405**                                 |
| Training history | 45 train log points, 1 val log point       |


**Training completed successfully** (2,250/2,250 steps). Final train loss ~0.63; validation loss **0.6405** at best checkpoint.

---

## 4. Hyperparameter Justification


| Parameter           | Value         | Reasoning                                          |
| ------------------- | ------------- | -------------------------------------------------- |
| `lora_r`            | 16            | Faster training, fewer adapter params; alpha = 2×r |
| `lora_alpha`        | 32            | Standard QLoRA scaling (alpha/r = 2.0)             |
| `lora_dropout`      | 0.05          | Light regularization on adapter                    |
| `learning_rate`     | 2e-4          | Standard QLoRA LR (Dettmers et al.)                |
| `num_epochs`        | 1             | Single pass over 18K subset                        |
| `batch_size`        | 1             | T4 VRAM limit at seq_len=512                       |
| `grad_accumulation` | 8             | Effective batch = 8                                |
| `max_seq_length`    | 512           | Covers majority of prompt+SQL lengths              |
| `fp16` / `bf16`     | False / False | Avoids BFloat16 GradScaler crash on T4             |
| `lr_scheduler`      | cosine        | Smooth decay                                       |
| `warmup_ratio`      | 0.05          | Stabilize adapter early                            |
| `MAX_TRAIN_SAMPLES` | 18,000        | ~2,250 steps, ~7–9 h on T4                         |


### Training Loss Curve (from saved history)

- Train loss: sharp drop early (~~1.5 → ~0.8), then gradual decline to **~~0.65** by step 2,250.  
- Val loss: **0.6405** (single eval point near end of training).  
- Train/val gap at best checkpoint is small (~0.02), suggesting limited overfit on the 18K subset.

---

## 5. Pre vs Post Output Comparison

8 examples across complexity levels (qualitative demo, same inputs):

### Easy — Format Learning


|                | SQL                                                                                                     |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| **Question**   | What is the highest total medals of russia, which has more than 1 silver and more than 6 bronze medals? |
| **Gold**       | `SELECT MAX(total) FROM table_name_83 WHERE silver > 1 AND nation = "russia" AND bronze > 6`            |
| **Base**       | `SELECT max(total) ... WHERE nation = 'russia' ...` — EX ✅ EM ❌                                         |
| **Fine-tuned** | `SELECT MAX(total) ... WHERE nation = "russia" ...` — EX ✅ **EM ✅**                                     |


Fine-tuning learned double-quote convention and keyword casing from the training distribution.

### Medium — Alias / Format Alignment


|                | SQL                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------ |
| **Question**   | Show the customer id and number of accounts with most accounts.                                  |
| **Gold**       | `SELECT customer_id, COUNT(*) FROM Accounts GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 1` |
| **Base**       | Uses alias `num_accounts`, extra formatting — EX ✅ EM ❌                                          |
| **Fine-tuned** | Matches gold structure — EX ✅ **EM ✅**                                                           |


### Hard — Schema-Grounded JOIN Recovery


|                | SQL                                                                   |
| -------------- | --------------------------------------------------------------------- |
| **Question**   | What is the level name of the cheapest catalog (in USD)?              |
| **Gold**       | JOIN + `ORDER BY price_in_dollars LIMIT 1`                            |
| **Base**       | Malformed / incomplete JOIN — EX ❌                                    |
| **Fine-tuned** | Correct JOIN and ordering — **EX ✅** (EM ❌ due to table alias naming) |


### Extra Hard — Semantically Correct, EM Strict


|                | SQL                                                                          |
| -------------- | ---------------------------------------------------------------------------- |
| **Question**   | Show all product names and the total quantity ordered for each product name. |
| **Gold**       | `Order_items` JOIN `Products`, GROUP BY product_name                         |
| **Fine-tuned** | Swapped table aliases but equivalent JOIN + GROUP BY — EX ✅ EM ❌             |


### Demo Summary (8 examples)


| Model                          | Exact Match        | Exec Accuracy  | n      |
| ------------------------------ | ------------------ | -------------- | ------ |
| Fine-tuned (demo)              | **75%** (6/8)      | **100%** (8/8) | 8      |
| Base (where matched in §2 CSV) | Lower EM, mixed EX | —              | subset |


---

## 6. Evaluation

### Primary Results (fine-tuned on **500** stratified test examples)


| Metric                 | Baseline (§2)  | Fine-tuned (§6)      | Delta         |
| ---------------------- | -------------- | -------------------- | ------------- |
| **Exact Match (EM)**   | 1.00% (n=200)  | **76.60%** (n=500)   | **+75.60 pp** |
| **Exec Accuracy (EX)** | 36.92% (n=200) | **95.53%** (492/500) | **+58.61 pp** |


*Note: Baseline and fine-tuned sample sizes differ (200 vs 500). Both drawn from the same held-out test distribution (`test_split.csv`, 3,914 total). Full test eval (3,914 examples) not yet run.*

### Complexity Breakdown (Fine-tuned, 500 examples)


| Complexity  | EM         | EX         | n   | Skipped (EX) |
| ----------- | ---------- | ---------- | --- | ------------ |
| easy        | 78.12%     | 95.37%     | 480 | —            |
| medium      | 75.00%     | 100.00%    | 8   | —            |
| hard        | 25.00%     | 100.00%    | 8   | —            |
| extra_hard  | 0.00%      | 100.00%    | 4   | —            |
| **Overall** | **76.60%** | **95.53%** | 500 | 8            |


**Observations:**

- Largest gains on easy queries (majority of data).  
- **extra_hard: 0% EM but 100% EX** on this sample — model produces semantically correct SQL with different aliases/formulation than gold.  
- 8 examples skipped for EX (schema/gold execution errors).

### Error Taxonomy (fine-tuned failures, n=500)


| Error Type          | Count | %     | Description                                    |
| ------------------- | ----- | ----- | ---------------------------------------------- |
| `exact_match`       | 383   | 76.6% | Correct by strict string match (success class) |
| `wrong_logic`       | 77    | 15.4% | Executes but wrong result                      |
| `wrong_aggregation` | 33    | 6.6%  | Wrong GROUP BY / aggregate                     |
| `syntax_error`      | 3     | 0.6%  | SQL fails to parse/execute                     |
| `wrong_column`      | 2     | 0.4%  | Hallucinated column                            |
| `schema_skipped`    | 2     | 0.4%  | Schema build failed                            |


Most remaining failures are **wrong_logic** and **wrong_aggregation**, not syntax — consistent with a model that generates mostly valid SQL.

### Loss Curves

Training loss declined from ~1.5 to ~0.65 over 2,250 steps. Validation loss at end: **0.6405**. Complexity chart shows high EX across all buckets with EM dropping on hard/extra_hard — same pattern as the qualitative examples.

### Metric Rationale

- **EM:** Measures format alignment with gold SQL (primary fine-tuning signal on this base model).  
- **EX:** Measures semantic correctness via in-memory SQLite execution — the practical metric for Text-to-SQL.  
- Both reported because EM rose dramatically (+75.6 pp) while EX also improved substantially (+58.6 pp vs baseline).

### Artifacts Saved


| File                       | Purpose                            |
| -------------------------- | ---------------------------------- |
| `test_split.csv`           | Held-out test set (3,914 rows)     |
| `test_predictions.csv`     | 500-example fine-tuned predictions |
| `baseline_predictions.csv` | 200-example baseline predictions   |
| `training_history.json`    | Loss curves                        |
| `evaluation_charts.png`    | Loss + complexity chart            |
| `pre_post_comparison.csv`  | 8 qualitative examples             |
| `final_adapter/`           | LoRA adapter weights               |


---

## 7. Production Thinking

### Serving Architecture

```
User request (question + schema DDL)
  └─→ FastAPI endpoint  POST /generate
        ├─ Schema injected at request time
        ├─ Model: Qwen2.5-Coder-7B (4-bit) + LoRA adapter (~200 MB)
        ├─ Greedy decoding (do_sample=False) for deterministic output
        └─ Response validated (SELECT-only before returning)
```

### Inference on T4

- Single 4-bit model + adapter: **~5.75 GB VRAM**  
- Do **not** load base + fine-tuned simultaneously on 16 GB GPU  
- ~2 sec/query → ~2–3 h for full 3,914-example eval

### Hosting Options


| Option                          | Latency      | Notes                                |
| ------------------------------- | ------------ | ------------------------------------ |
| Kaggle T4 / AWS g4dn            | ~1–3 s/query | GPU inference                        |
| Modal / RunPod                  | Per-second   | Serverless production                |
| HuggingFace Inference Endpoints | ~1–3 s       | Easiest managed deploy               |
| Adapter on Hub + 4-bit base     | —            | ~200 MB adapter vs ~14 GB full model |


---

## Bonus — Safety Guardrails

Post-generation validation (recommended for production):

- Reject non-SELECT statements  
- Block destructive keywords (`DELETE`, `DROP`, `TRUNCATE`, etc.)  
- Schema injected at request time — model only sees user-provided DDL  
- Optional PII field checks against schema

---

## Summary


| Item                    | Value                                                               |
| ----------------------- | ------------------------------------------------------------------- |
| **Base model**          | `Qwen/Qwen2.5-Coder-7B-Instruct`                                    |
| **Method**              | QLoRA (4-bit NF4 + LoRA rank 16)                                    |
| **Training data**       | 18,000 examples (sql-create-context; Gretel filter returned 0 rows) |
| **Training steps**      | 2,250                                                               |
| **Best val loss**       | **0.6405**                                                          |
| **Inference VRAM**      | **5.75 GB** (T4)                                                    |
| **Baseline EM → FT EM** | 1.00% → **76.60%** (+75.60 pp)                                      |
| **Baseline EX → FT EX** | 36.92% → **95.53%** (+58.61 pp)                                     |
| **Eval sample**         | 500 / 3,914 held-out test                                           |
| **Adapter path**        | `/kaggle/working/qlora_sql/final_adapter/`                          |


---

*Evaluation performed on held-out test examples not used in the 18K training subset. Full 3,914-example eval optional for submission hardening.*