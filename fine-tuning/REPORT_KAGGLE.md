# AI League PS5 — Text-to-SQL Fine-Tuning (Kaggle Run)
## Project Report — Qwen2.5-Coder-7B + QLoRA on T4 GPU

**Author:** Kirti  
**Date:** June 2026  
**Problem Statement:** Problem 1 — Text-to-SQL (Code Generation)  
**Hardware:** Kaggle Tesla T4 × 2, 15.6GB VRAM each  
**Framework:** PyTorch + HuggingFace TRL + bitsandbytes QLoRA  
**Status:** Training cut off at step 601/1500 — evaluation cells did not run

> **Note:** This is the Kaggle-based run using the larger 7B model. The complete, evaluated run
> is documented in `REPORT.md` (local MacBook M4 Air, 3B MLX model). Both runs share the same
> dataset pipeline, prompt format, and evaluation methodology.

---

## Table of Contents

1. [Dataset Cleaning and Build](#1-dataset-cleaning-and-build)
2. [Model Choice and Baseline Benchmark](#2-model-choice-and-baseline-benchmark)
3. [Training Strategy](#3-training-strategy)
4. [Hyperparameter Justification](#4-hyperparameter-justification)
5. [Training Progress (Partial)](#5-training-progress-partial)
6. [Evaluation](#6-evaluation)
7. [Production Thinking](#7-production-thinking)

---

## 1. Dataset Cleaning and Build

### Source

**`b-mc2/sql-create-context`** — 78,577 natural-language → SQL pairs.  
**`gretelai/synthetic_text_to_sql`** — 100K synthetic pairs (targeted: challenging + moderate complexity).

> **Note:** The Gretel dataset returned 0 rows in this run. The dataset's `sql_complexity`
> column field names appear to have changed upstream. Only sql-create-context was used.

### Prompt Format

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

| Step | What was removed | Result |
|---|---|---|
| Null / empty rows | Missing question, context, or SQL | 78,577 → 78,577 (0 removed) |
| SQL parse validation | Unparseable strings (via sqlparse) | 78,577 → 78,577 (0 removed) |
| Non-SELECT filter | INSERT / UPDATE / DELETE / DROP / DDL | 78,577 → 78,577 (0 removed) |
| Deduplication | Exact question duplicates (MD5 hash) | 78,577 → **78,262** (315 removed) |

### Complexity Distribution

| Class | Rule | Count | % |
|---|---|---|---|
| `easy` | Simple SELECT/WHERE, single table | 75,287 | 96.2% |
| `extra_hard` | Multi-JOIN or JOIN + aggregation | 1,057 | 1.3% |
| `medium` | GROUP BY / HAVING / subquery / UNION | 1,045 | 1.3% |
| `hard` | Single JOIN | 873 | 1.1% |

### Train / Val / Test Split (Stratified by Complexity)

| Split | Size | Purpose |
|---|---|---|
| Train | 68,087 (full) / 3,000 (subset used) | Fine-tuning |
| Validation | 6,261 | Early stopping |
| Test | 3,914 | Final evaluation (not run — see §6) |

---

## 2. Model Choice and Baseline Benchmark

### Model: `Qwen/Qwen2.5-Coder-7B-Instruct` (4-bit NF4 QLoRA)

| Criterion | Justification |
|---|---|
| **Size (7B)** | 4× larger capacity than 3B variant. Better at multi-table SQL and complex aggregations. |
| **Coder family** | Pre-trained on SQL-heavy code corpus. Strong SQL baseline without any fine-tuning. |
| **Instruction-tuned** | Responds to `### Task / ### SQL` prompt format without additional system prompt engineering. |
| **4-bit NF4 QLoRA** | Reduces VRAM from ~28GB (FP16 7B) to ~8-10GB during training — fits on a 16GB T4. |
| **Apache 2.0 license** | Production-usable without restriction. |

**Trainable parameters:**
- LoRA rank 64, alpha 128, targeting all projection layers
- Trainable: **161,480,704** parameters (3.58% of total 4,514,452,992)

### Baseline Results (untuned model, 200 test examples)

| Metric | Score | Notes |
|---|---|---|
| **Exact Match (EM)** | 1.00% | Low due to quote style / keyword casing mismatch |
| **Exec Accuracy (EX)** | 38.97% (195/200 evaluated) | Lower than expected — SQLite cannot parse MySQL-style DDL (backticks, INT(11), etc.) without sanitization. 5 examples skipped due to schema build errors. |

| Complexity | EM | EX | n |
|---|---|---|---|
| easy | 0.52% | 38.95% | 194 |
| medium | 33.33% | 66.67% | 3 |
| hard | 0.00% | 0.00% | 3 |

> **Why is EX only 38.97% vs 88.66% in the local MLX run?**  
> The Kaggle run does not include MySQL→SQLite DDL sanitization (backticks → double quotes,
> `INT(11)` → `INTEGER`, `VARCHAR(255)` → `TEXT`, etc.). Most schema builds silently fail,
> causing SQLite to reject the DB — those examples count as EX failures. The local MLX run
> added `sanitize_schema()` which resolved this, raising EX from ~39% to ~89% for the untuned model.

---

## 3. Training Strategy

### Method: Supervised Fine-Tuning (SFT) with QLoRA

**Why SFT:** Labeled NL→SQL pairs with gold answers — supervised fine-tuning is the correct default.

**Why QLoRA over full fine-tuning:**
- Full fine-tuning of 7B params requires ~56GB for weights + Adam optimizer states. Impossible on T4 (16GB).
- QLoRA: 4-bit NF4 base weights (frozen) + FP16 LoRA adapter weights (trainable). Peak VRAM: ~10-12GB.
- Adapter weights (~200MB) are shareable independently of the 14GB base model.

**Why not prompt tuning / prefix tuning:**
- LoRA has stronger expressiveness for generative code tasks and trains faster.
- Prompt tuning does not modify attention weights — insufficient for schema-faithful SQL generation.

**Loss function:** Cross-entropy over the full sequence. TRL's `SFTTrainer` masks the prompt prefix and computes loss only on SQL tokens (completion-only masking). The model learns SQL generation, not prompt memorization.

**Target modules:** All linear projection layers — `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Targeting all projections gives maximum adapter capacity for schema-faithful generation.

---

### TRL — The Training Library

**TRL (Transformer Reinforcement Learning)** is HuggingFace's library for post-training LLMs.  
Version used in this run: **trl 1.6.0**

TRL was chosen over vanilla HuggingFace `Trainer` because:
- `SFTTrainer` natively integrates with PEFT/LoRA — no manual adapter wrapping needed
- Built-in completion-only masking: prompt tokens get label `-100` (ignored in loss), SQL answer tokens carry full cross-entropy loss
- Automatic dataset collation, packing, and group-by-length batching for efficiency

**Key TRL classes used:**

| Class | Role |
|---|---|
| `SFTConfig` | Subclass of `TrainingArguments`. Controls all training loop settings: LR, epochs, batch size, scheduler, logging, checkpointing. |
| `SFTTrainer` | Orchestrates the training loop. Wraps the model with the LoRA adapter (`peft_config` arg), handles evaluation, and saves the best checkpoint. |

**How `SFTConfig` and `SFTTrainer` work together:**

```python
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type='cosine',
    warmup_steps=50,          # replaces deprecated warmup_ratio
    bf16=True,                # Qwen2.5 is BFloat16 native
    eval_strategy='steps',
    eval_steps=100,
    ...
)

trainer = SFTTrainer(
    model=base_model,          # frozen 4-bit base
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    peft_config=lora_config,   # SFTTrainer injects LoRA here
)
```

**TRL API evolution — issues encountered and fixed:**

TRL changed its API significantly across minor versions. Several errors were hit and resolved:

| TRL Version | Breaking Change | Fix Applied |
|---|---|---|
| ≥ 0.11.0 | `max_seq_length` moved from `SFTConfig` → `SFTTrainer` | Moved param to `SFTTrainer(max_seq_length=...)` |
| ≥ 0.12.0 | `max_seq_length` removed from `SFTTrainer` entirely | Set `tokenizer.model_max_length` directly; remove from both |
| ≥ 0.12.0 | `warmup_ratio` deprecated | Replaced with `warmup_steps=50` in `SFTConfig` |
| 1.6.0 | `dataset_text_field` moved back to `SFTTrainer` | Updated signature accordingly |

**Dtype issue — `bf16=True` was critical:**  
Qwen2.5's weights are natively stored in BFloat16. Setting `fp16=True` (the old default) caused:
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```
The fix was `fp16=False, bf16=True` in `SFTConfig`. The FP16 gradient scaler doesn't support BFloat16 tensors; using `bf16=True` routes gradients through the correct BF16 path.

---

## 4. Hyperparameter Justification

| Parameter | Value | Reasoning |
|---|---|---|
| `lora_rank` (r) | 64 | Higher rank = more adapter capacity for 7B model. Provides 161M trainable params (3.58%). |
| `lora_alpha` | 128 | 2×r convention. Scale factor = alpha/r = 2.0. |
| `lora_dropout` | 0.05 | Light dropout on adapter to reduce overfit on repeated schema patterns. |
| `learning_rate` | 2e-4 | Standard QLoRA LR from Dettmers et al. Higher than MLX run (1e-4) because QLoRA uses quantized gradients. |
| `num_train_epochs` | 2 | With 3,000 training samples → 1,500 steps total. |
| `batch_size` | 1 (per device) | 16GB T4 VRAM limit at seq_len=256 with 4-bit base + FP16 grads. |
| `grad_accumulation` | 4 | Effective batch = 4. Larger accum = more stable gradient estimates. |
| `lr_scheduler` | cosine | Smooth LR decay prevents late-training instability. |
| `warmup_steps` | 50 | ~3.3% warmup for adapter weights to stabilise before full LR. |
| `max_seq_length` | 256 | Truncated for speed on T4. 95th pct of prompts fits in 512 but 256 is sufficient for most easy queries. |
| `bf16` | True | Qwen2.5 uses BFloat16 natively. Using bf16=True avoids dtype mismatch with FP16 gradient scaler. |
| `dataloader_num_workers` | 0 | Avoids multiprocessing issues in Kaggle's committed environment. |

---

## 5. Training Progress (Partial)

> **Training was cut off at step 601 of 1500 when the Kaggle session ended.**  
> The following data is from the training log up to step 601.

### Loss Curve Data (steps 100–500)

| Step | Train Loss | Val Loss | Token Accuracy | Notes |
|---|---|---|---|---|
| 100 | 0.717 | 0.870 | 80.6% | Strong early descent |
| 200 | 0.699 | 0.800 | 81.4% | Val loss dropping fast |
| 300 | 0.698 | 0.776 | 81.7% | |
| 400 | 0.649 | 0.774 | 81.8% | Val loss plateauing |
| 500 | 0.692 | **0.749** | 82.1% | Still improving, no overfit |

**Diagnosis:** Val loss was still decreasing at step 500 (the last checkpoint before cutoff). Unlike the local MLX run (which overfit at iter 300), the 7B model on 3,000 samples was still converging cleanly. Given the val loss trajectory, the model likely would have reached ~0.72-0.73 val loss by step 1500.

**Comparison with local MLX run:**

| | Kaggle 7B (partial) | Local MLX 3B (complete) |
|---|---|---|
| Val loss at step 100 | 0.870 | — (MLX uses iters not steps) |
| Best val loss observed | 0.749 (step 500) | **0.763 (iter 200)** |
| Training stopped | Step 601 (session ended) | Iter 200 (early stop, best checkpoint) |
| Overfitting observed | Not yet | Yes, at iter 300 |

---

## 6. Evaluation

> **The training session ended before evaluation cells could run.**  
> Final evaluation results (EM, EX, complexity breakdown, error taxonomy) are **not available** for this run.

### What we know:
- Training was converging well at step 500 (val loss 0.749, token accuracy 82.1%)
- The base model baseline was EM=1.00%, EX=38.97% (lower EX due to missing schema sanitization)
- Based on the local 3B run's trajectory (EM 3%→70% with 2K samples), the 7B model with 3K samples and longer training likely would have achieved **higher EM and EX** — particularly on hard and extra_hard queries where the 3B model struggled

### Schema Sanitization Note

The baseline EX of 38.97% in this run is artificially low. The evaluation code does not convert MySQL DDL to SQLite-compatible syntax before building the in-memory database. Adding the `sanitize_schema()` function from the local MLX run would bring the untuned EX up to ~88-89%, consistent with the local run's baseline.

---

## 7. Production Thinking

The production architecture is identical to the local MLX run:

```
User request (question + schema DDL)
  └─→ FastAPI endpoint  POST /generate
        ├─ Schema injected at request time
        ├─ Model: Qwen2.5-Coder-7B (4-bit QLoRA) + LoRA adapter (~200MB)
        ├─ Greedy decoding (temperature=0) for deterministic output
        └─ Response validated (must be SELECT before returning)
```

**7B vs 3B in production:**

| | 3B (local MLX) | 7B (QLoRA T4) |
|---|---|---|
| Adapter size | ~25 MB | ~200 MB |
| Inference latency | ~0.5–1s (M4 Air) | ~1–3s (T4 GPU) |
| Memory required | ~5.1GB (MLX 4-bit) | ~8–10GB (bitsandbytes 4-bit) |
| Expected quality | Good on easy/medium | Better on hard/extra_hard |
| Hosting cost | $0 local | ~$0.53/hr on AWS g4dn.xlarge |

A FastAPI serving stub was written to `serve.py` with:
- Non-SELECT output rejection
- Blocked keyword filtering (DELETE, DROP, etc.)
- Schema grounding (schema injected per request, not baked into model)

---

## Summary

| Item | Value |
|---|---|
| **Base model** | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| **Method** | QLoRA (r=64, alpha=128, 4-bit NF4) |
| **Framework** | PyTorch + HuggingFace TRL 1.6.0 (`SFTConfig` + `SFTTrainer`) + bitsandbytes |
| **Training data** | 3,000 examples from `b-mc2/sql-create-context` |
| **Training status** | Incomplete — cut off at step 601/1500 |
| **Best val loss observed** | 0.749 (step 500, still improving) |
| **Trainable params** | 161.5M / 4,514.5M (3.58%) |
| **Baseline EM** | 1.00% (200 examples) |
| **Baseline EX** | 38.97% (without schema sanitization; ~89% with it) |
| **Final EM / EX** | Not available — training session ended before evaluation |

---

## Lessons Learned

1. **Gretel dataset field names changed** — `sql_complexity` column values may no longer be 'challenging'/'moderate'. Should use `.value_counts()` first to inspect actual values before filtering.

2. **Schema sanitization is essential** — Without MySQL→SQLite conversion, execution accuracy is artificially low (~39%). The `sanitize_schema()` utility from the local MLX run must be ported to any Kaggle eval cell.

3. **Kaggle session limits** — A 10+ hour training run on a single T4 session is risky. Strategies to mitigate:
   - Use `Save & Run All` (committed mode) to avoid session timeouts
   - Checkpoint frequently (`save_every=100`) so work is not lost
   - Consider P100 over T4 for faster throughput

4. **7B vs 3B trade-off** — The 7B model's val loss was still decreasing at step 500 with no overfitting, unlike the 3B which overfit at iter 300. For a full training run, the 7B would likely significantly outperform the 3B, especially on hard/extra_hard queries.

---

*Generated for AI League PS5. Kaggle run (7B) — partial. See `REPORT.md` for the complete local MLX run (3B) with full evaluation results.*
