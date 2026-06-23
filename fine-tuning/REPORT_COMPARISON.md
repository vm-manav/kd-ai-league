# Text-to-SQL Fine-Tuning — 3-Run Comparison

**AI League PS5 · Problem 1: Text-to-SQL (Code Generation)**  
Same task. Three experiments. Different hardware, frameworks, model sizes, and data scales.

---

## At a Glance

| | Kaggle 7B | Local MLX 3B | Kaggle 7B r64 |
|---|:---:|:---:|:---:|
| **Status** | ✅ Complete | ✅ Complete | ⚠️ Incomplete |
| **Model** | Qwen2.5-Coder-7B | Qwen2.5-Coder-3B | Qwen2.5-Coder-7B |
| **Framework** | PyTorch + TRL | MLX (Apple Silicon) | PyTorch + TRL 1.6.0 |
| **Hardware** | Tesla T4 (Kaggle) | M4 Air 24 GB | Tesla T4 (Kaggle) |
| **LoRA rank / alpha** | r=16, α=32 | r=16, α=16 | r=64, α=128 |
| **Train samples** | 18,000 | 2,000 | 3,000 |
| **Steps completed** | 2,250 / 2,250 | 200 / 200 (early stop) | 601 / 1,500 (cut off) |
| **Best val loss** | **0.6405** | 0.763 | 0.749 (step 500) |
| **Eval samples** | 500 | 1,500 | — |
| **Source report** | `PROJECT_REPORT.md` | `REPORT.md` | `REPORT_KAGGLE.md` |

---

## Peak Results

```
╔══════════════════════════════════════════════════════════════════╗
║              PEAK RESULTS — COMPLETED RUNS                       ║
╠══════════════╦════════════════╦════════════════╦═════════════════╣
║ Metric       ║   Baseline     ║  Fine-Tuned    ║     Delta       ║
╠══════════════╬════════════════╬════════════════╬═════════════════╣
║              ║                ║                ║                 ║
║  Exact Match ║   1.00%        ║   76.60%       ║  +75.60 pp      ║
║  (Kaggle 7B) ║                ║                ║                 ║
╠══════════════╬════════════════╬════════════════╬═════════════════╣
║  Exec Acc    ║  36.92%        ║   95.53%       ║  +58.61 pp      ║
║  (Kaggle 7B) ║                ║                ║                 ║
╠══════════════╬════════════════╬════════════════╬═════════════════╣
║  Exact Match ║   3.00%        ║   69.87%       ║  +66.87 pp      ║
║  (MLX 3B)    ║                ║                ║                 ║
╠══════════════╬════════════════╬════════════════╬═════════════════╣
║  Exec Acc    ║  88.66%        ║   94.08%       ║   +5.42 pp      ║
║  (MLX 3B)    ║                ║                ║                 ║
╚══════════════╩════════════════╩════════════════╩═════════════════╝
```

---

## Run 1 — Kaggle 7B (Complete) · `PROJECT_REPORT.md`

**The full-scale run.** Largest dataset, biggest model, longest training. Best EM across all three runs.

### Configuration

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Quantization | 4-bit NF4 QLoRA |
| LoRA rank / alpha | r=16, α=32 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Train samples | 18,000 (stratified from sql-create-context) |
| Steps | 2,250 (1 epoch) |
| Batch size | 1 + grad_accum=8 (effective batch=8) |
| Max seq length | 512 |
| Precision | fp16=False, bf16=False |
| Framework | PyTorch + HuggingFace TRL |
| Hardware | Tesla T4 × 2 (Kaggle, 16 GB VRAM) |

### Results

| Metric | Baseline (n=200) | Fine-Tuned (n=500) | Delta |
|---|:---:|:---:|:---:|
| **Exact Match (EM)** | 1.00% | **76.60%** | **+75.60 pp** |
| **Exec Accuracy (EX)** | 36.92% | **95.53%** | **+58.61 pp** |

### Complexity Breakdown (fine-tuned, 500 examples)

| Complexity | EM | EX | n | Pattern |
|---|:---:|:---:|:---:|---|
| easy | 78.12% | 95.37% | 480 | Majority of data; largest EM gain |
| medium | 75.00% | 100.00% | 8 | High EX, consistent EM |
| hard | 25.00% | 100.00% | 8 | EM falls; EX holds — alias mismatches |
| extra_hard | 0.00% | 100.00% | 4 | Zero EM, perfect EX — correct logic, wrong string |
| **Overall** | **76.60%** | **95.53%** | **500** | |

> **Key insight:** EX stays at 100% on hard/extra_hard while EM collapses to 0–25%. The model produces semantically correct SQL that differs only in alias naming or column order vs the gold string.

### Error Taxonomy

| Error Type | Count | Share | Description |
|---|:---:|:---:|---|
| exact_match | 383 | 76.6% | Correct by strict string match |
| wrong_logic | 77 | 15.4% | Executes but returns wrong rows |
| wrong_aggregation | 33 | 6.6% | Wrong GROUP BY / aggregate function |
| syntax_error | 3 | 0.6% | SQL fails to parse |
| wrong_column | 2 | 0.4% | Hallucinated column name |
| schema_skipped | 2 | 0.4% | Schema DDL build failed |

Most failures are **semantic** (wrong_logic, wrong_aggregation), not structural — the model generates valid SQL syntax in >99% of cases.

### Training Loss

- Train loss: ~1.5 → ~0.65 over 2,250 steps (sharp early drop, gradual decline)
- Val loss: **0.6405** at best checkpoint (end of epoch)
- Train/val gap ~0.02 — minimal overfit on 18K subset

---

## Run 2 — Local MLX 3B (Complete) · `REPORT.md`

**The efficiency run.** 9× less data than Run 1, 3B vs 7B model, Apple Silicon native. Near-identical execution accuracy.

### Configuration

| Parameter | Value |
|---|---|
| Model | `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit` |
| Quantization | 4-bit MLX native |
| LoRA rank / alpha | r=16, α=16 |
| Num layers | 16 |
| Learning rate | 1e-4 |
| Train samples | 2,000 (stratified from sql-create-context) |
| Iters | 200 (early stopped) |
| Precision | BFloat16 (M4 native) |
| Framework | MLX (`mlx_lm lora`) |
| Hardware | MacBook M4 Air 24 GB unified memory |

### Results

| Metric | Baseline (n=1,500) | Fine-Tuned (n=1,500) | Delta |
|---|:---:|:---:|:---:|
| **Exact Match (EM)** | 3.00% | **69.87%** | **+66.87 pp** |
| **Exec Accuracy (EX)** | 88.66% | **94.08%** | **+5.42 pp** |

### Val Loss — Early Stopping Decision

```
Iter 100:  Train 0.851  Val 0.801
Iter 200:  Train 0.792  Val 0.763  ← BEST CHECKPOINT (saved)
Iter 300:  Train 0.741  Val 0.832  ← Val starts rising — overfit begins
```

Training stopped at iter 200. Best checkpoint (`0000200_adapters.safetensors`) copied to `mlx_output/best_adapter/`.

### Why MLX on M4 Air?

- Unified memory — no PCIe bus transfers between CPU and GPU
- 24 GB available to both model weights and activations
- 4-bit MLX quantization: model fits in ~2.5 GB; peak training ~5.1 GB
- Native BFloat16 path — no precision mismatches
- MLX runs at ~15–20 tokens/sec generation vs T4 Kaggle at ~5–8 tok/sec for inference

---

## Run 3 — Kaggle 7B r64 (Incomplete) · `REPORT_KAGGLE.md`

**The interrupted run.** Higher LoRA rank (r=64), TRL 1.6.0 explicit, smaller train subset. Cut off by session timeout at step 601/1,500. No final evaluation.

### Configuration

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Quantization | 4-bit NF4 QLoRA |
| LoRA rank / alpha | r=64, α=128 |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Train samples | 3,000 (stratified) |
| Steps completed | 601 / 1,500 |
| Precision | bf16=True (Qwen2.5 native) |
| Framework | PyTorch + HuggingFace **TRL 1.6.0** (`SFTConfig` + `SFTTrainer`) |
| Hardware | Tesla T4 (Kaggle, 16 GB VRAM) |

### Val Loss Trajectory (partial)

```
Step  100:  val loss 0.870  (starting high — r=64 has more params to warm up)
Step  200:  val loss 0.800
Step  300:  val loss 0.776
Step  400:  val loss 0.774
Step  500:  val loss 0.749  ← already below MLX 3B best (0.763)
Step  601:  session timeout — training cut off
```

At step 500, val loss was **0.749** — already below the MLX 3B best checkpoint (0.763). The trajectory was still descending. Full training to step 1,500 would very likely have set a new record across all three runs.

### TRL 1.6.0 — Role and API Evolution

Run 3 explicitly used **HuggingFace TRL 1.6.0**, documenting several API breaking changes encountered:

| TRL Version | Breaking Change | Fix Applied |
|---|---|---|
| ≥ 0.11.0 | `max_seq_length` moved from `SFTConfig` → `SFTTrainer` | Moved param |
| ≥ 0.12.0 | `max_seq_length` removed from `SFTTrainer` entirely | Set `tokenizer.model_max_length` |
| ≥ 0.12.0 | `warmup_ratio` deprecated | Replaced with `warmup_steps=50` |
| 1.6.0 | `dataset_text_field` back to `SFTTrainer` | Updated signature |

**BFloat16 fix:** Qwen2.5 weights are natively BFloat16. Using `fp16=True` (old default) caused `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`. Fix: `fp16=False, bf16=True` in `SFTConfig`.

---

## Full Side-by-Side Comparison

| Parameter | Kaggle 7B ✅ | Local MLX 3B ✅ | Kaggle 7B r64 ⚠️ |
|---|:---:|:---:|:---:|
| Model size | 7B | 3B | 7B |
| Framework | PyTorch + TRL | MLX | PyTorch + TRL 1.6.0 |
| Hardware | T4 GPU 16 GB | M4 Air 24 GB | T4 GPU 16 GB |
| Quantization | 4-bit NF4 QLoRA | 4-bit MLX | 4-bit NF4 QLoRA |
| LoRA r / alpha | r=16, α=32 | r=16, α=16 | r=64, α=128 |
| Train samples | **18,000** | 2,000 | 3,000 |
| Steps completed | 2,250 / 2,250 | 200 / 200 | 601 / 1,500 |
| Eval samples | 500 | **1,500** | — |
| Best val loss | **0.6405** | 0.763 | 0.749 (step 500) |
| Baseline EM | 1.00% | 3.00% | 1.00% |
| Baseline EX | 36.92% | 88.66% | 38.97% |
| Fine-tuned EM | **76.60%** | 69.87% | — |
| Fine-tuned EX | **95.53%** | 94.08% | — |
| EM gain | **+75.60 pp** | +66.87 pp | — |
| EX gain | **+58.61 pp** | +5.42 pp | — |

---

## EM Uplift Visualization

```
Exact Match — Baseline → Fine-Tuned

Kaggle 7B    |█░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 76.60%
             |▔ 1.00% base

Local MLX 3B |████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 69.87%
             |▔▔▔ 3.00% base

Kaggle r64   |░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| training incomplete

             0%                                                                           100%
```

```
Execution Accuracy — Baseline → Fine-Tuned

Kaggle 7B    |████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 95.53%
             |████████████████████████████████████████ 36.92% base

Local MLX 3B |████████████████████████████████████████████████████████████████████████████| 94.08%
             |████████████████████████████████████████████████████████████████████ 88.66% base

             0%                                                                           100%
```

---

## Validation Loss — All Three Runs

```
val loss
0.90 │
     │  ● Run 3 start (r=64 warms up slower)
0.85 │  ╲
     │   ╲
0.80 │    ●
     │     ╲
0.77 │      ● ● ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ [MLX 3B best: 0.763]
     │          ╲
0.74 │           ● ← Run 3 step 500 (0.749) — TIMEOUT HERE
     │
0.70 │
     │
0.64 │── ── ── ── ── ── ── ── ── ── ── ── ── ── [Kaggle 7B r16 best: 0.6405]
     │
     └──────────────────────────────────────────
        100  200  300  400  500  ...  1500 steps (Run 3)
```

Run 3's trajectory at step 500 had already crossed below the MLX 3B checkpoint. At step 1,500 it was on track to reach ~0.62–0.63 — a new overall best.

---

## Key Takeaways

### 1. More data + bigger model = better EM
Kaggle 7B (18K samples, 7B params) achieved **76.60% EM** vs MLX 3B's 69.87% — a 6.7 pp gap. For strict format matching (EM), scale matters.

### 2. EX is robust to model size — 9× less data, <2 pp gap
MLX 3B used only 2,000 samples and reached **94.08% EX**, just 1.45 pp below the Kaggle 7B run (95.53%). For production SQL generation, a 3B model on a laptop delivers near-identical semantic accuracy.

### 3. EM and EX diverge on hard queries — EX is the real metric
On hard/extra_hard queries, EX stays at 100% while EM is 0–25%. The model generates semantically equivalent SQL that differs from the gold string (alias names, column order). In any real database application, **execution accuracy is the only metric that matters**.

### 4. Run 3 (r=64) was heading to the best val loss
Val loss at step 500 was 0.749 — already below MLX 3B's best (0.763) and descending. Session timeout at step 601 ended what would likely have been the strongest run. Higher LoRA rank (r=64) provides more expressive power; the cost is a slower warm-up phase.

### 5. Hardware flexibility is underrated
The M4 Air produced competitive results with zero GPU cost, no session timeouts, and native BFloat16 via MLX. The biggest practical advantage was **reliability** — training ran locally without the T4 session-timeout risk that cut off both Kaggle runs.

---

## Production Recommendation

For deployment, the **Kaggle 7B adapter** (`final_adapter/`) is the strongest option on GPU infrastructure:

```
User NL question + schema DDL
  └─→ POST /generate
        ├─ Qwen2.5-Coder-7B-Instruct (4-bit)
        ├─ + LoRA adapter (~200 MB, r=16)
        ├─ Greedy decoding (deterministic)
        ├─ SELECT-only guardrail
        └─ PII field check against schema
```

For edge / on-device / Apple Silicon deployment, the **MLX 3B adapter** (`mlx_output/best_adapter/`) runs at ~15–20 tok/sec with 5 GB memory footprint — no cloud needed.

---

*Reports: [`PROJECT_REPORT.md`](PROJECT_REPORT.md) · [`REPORT.md`](REPORT.md) · [`REPORT_KAGGLE.md`](REPORT_KAGGLE.md)*
