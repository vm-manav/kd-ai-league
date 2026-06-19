# Fine-Tuning — Text-to-SQL (AI League PS5)

**Task:** Natural language question → executable SQL query  
**Model:** `Qwen/Qwen2.5-Coder-7B-Instruct` + QLoRA (4-bit NF4)  
**Platform:** Kaggle (T4 / P100 GPU, 16GB VRAM)  

---

## Project structure

```
fine-tuning/
├── README.md                        ← you are here
├── requirements.txt                 ← all pip dependencies
├── configs/
│   └── qlora.yaml                   ← full hyperparameter config with justifications
├── src/
│   ├── data_pipeline.py             ← load / clean / format / split datasets
│   ├── evaluate.py                  ← EM + execution accuracy + error taxonomy
│   └── inference.py                 ← generate_sql(), batch_generate_sql(), FastAPI stub
├── notebooks/
│   └── kaggle_master.ipynb          ← THE submission notebook (all 7 mandatory steps)
├── data/                            ← local cache (gitignored)
└── outputs/                         ← model checkpoints, predictions, plots (gitignored)
```

---

## Datasets

| Source | HuggingFace ID | Size used | Role |
|--------|---------------|-----------|------|
| sql-create-context | `b-mc2/sql-create-context` | ~74K (post-clean) | Primary — schema-aware backbone |
| Gretel Synthetic | `gretelai/synthetic_text_to_sql` | 15K (challenging+moderate only) | Secondary — hard-query coverage |

**Why not Spider?** Spider's HuggingFace version doesn't include CREATE TABLE schemas inline. Execution accuracy requires in-memory SQLite, which needs the DDL — sql-create-context includes it directly.

---

## Evaluation metrics

| Metric | What it measures |
|--------|-----------------|
| **Exact Match (EM)** | Normalized string equality after keyword casing + whitespace collapse |
| **Execution Accuracy (EX)** | Run gold + predicted SQL against in-memory SQLite, compare result sets |
| **Complexity Breakdown** | EM + EX per bucket: easy / medium / hard / extra_hard |
| **Error Taxonomy** | Classify failures: syntax_error / wrong_table / wrong_column / wrong_aggregation / wrong_logic |

EM is strict but brittle. EX is the real measure — same result set = correct answer, even if written differently.

---

## How to run on Kaggle

### 1. Upload this repo as a Kaggle dataset or fork directly

```bash
# If using Kaggle CLI
kaggle datasets create -p ./fine-tuning
```

### 2. Open `notebooks/kaggle_master.ipynb` in Kaggle

- Enable GPU accelerator (T4 × 2 or P100)
- Run all cells top to bottom
- Total training time: ~10–11 hrs on T4 (1 epoch, full dataset)
- For a quick smoke test: set `MAX_TRAIN_SAMPLES = 2000` in the training cell

### 3. What gets saved to `/kaggle/working/qlora_sql/`

| File | Contents |
|------|----------|
| `final_adapter/` | LoRA adapter weights + tokenizer |
| `baseline_predictions.csv` | Base model predictions on 200 test examples |
| `test_predictions.csv` | Fine-tuned model predictions on full test set |
| `pre_post_comparison.csv` | Side-by-side examples across all complexity levels |
| `evaluation_charts.png` | Loss curves + complexity breakdown bar chart |
| `training_history.json` | Full step-by-step loss log |
| `serve.py` | FastAPI production serving stub |
| `eda.png` | Dataset EDA plots |

---

## Hyperparameter decisions

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `lora_r` | 64 | Higher rank = more adapter capacity for schema-faithful SQL generation |
| `lora_alpha` | 128 | Conventional 2×r scaling factor |
| `learning_rate` | 2e-4 | Standard QLoRA LR (Dettmers et al.); sweep tried 1e-4/2e-4/5e-4 |
| `epochs` | 1 | Full 74K dataset; 1 epoch ≈ 10h on T4. Val loss monitored for overfit |
| `batch_size` | 1 + grad_accum 8 | Effective batch = 8; VRAM limit on T4 at seq_len=512 |
| `max_seq_length` | 512 | 95th pct of prompts fit; longer sequences OOM on T4 |
| `lr_scheduler` | cosine | Smooth decay; prevents late-training instability |
| `warmup_ratio` | 0.05 | 5% steps for adapter weights to stabilise |

---

## Prompt format

```
### Task
Generate a SQL query to answer the following question.

### Database Schema
CREATE TABLE employees (id INT, name TEXT, salary FLOAT, dept_id INT);
CREATE TABLE departments (id INT, name TEXT);

### Question
What is the average salary per department?

### SQL
SELECT d.name, AVG(e.salary) FROM employees e JOIN departments d ON e.dept_id = d.id GROUP BY d.name
```

The loss is computed **only on the SQL tokens** (not the prompt prefix), so the model learns SQL generation exclusively.

---

## Mandatory steps map

| # | Step | Where in notebook |
|---|------|-------------------|
| 1 | Dataset cleaning and build | §1 — cells 5–9 |
| 2 | Model choice and baseline | §2 — cells 10–14 |
| 3 | Training strategy | §3 — cell 15 |
| 4 | Hyperparameter justification | §4 — cell 16 (markdown table + training args) |
| 5 | Pre vs post comparison | §5 — cell 18 |
| 6 | Evaluation + loss curves | §6 — cells 19–22 |
| 7 | Production thinking | §7 — cells 23–24 |

---

## Production deployment (Step 7 summary)

The fine-tuned adapter (~200MB) is published to HuggingFace Hub separately from the 14GB base model. At inference:

```python
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb_config)
model = PeftModel.from_pretrained(model, "your-username/qwen25-coder-7b-text2sql")
```

The FastAPI endpoint (`serve.py`) accepts `{question, schema}` and returns `{sql}`. Schema is injected at request time — the model is not tied to any specific database.

**Estimated inference latency:** 1–3 seconds per query on T4 GPU (4-bit quantized).  
**Hosting cost:** ~$0.53/hr on AWS g4dn.xlarge; $0 during idle on serverless (Modal/RunPod).

---

## Bonus: Safety guardrails

Three guardrail layers:
1. **Training data:** safety refusal examples added for out-of-domain questions, destructive SQL, and PII requests
2. **Post-generation validation:** reject any output that is not a SELECT statement or contains blocked keywords
3. **Schema grounding:** verify all referenced tables exist in the provided schema before returning the query
