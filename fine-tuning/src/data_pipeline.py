"""
Data pipeline for Text-to-SQL fine-tuning.

Sources:
  - b-mc2/sql-create-context   (primary, 78K, schema-aware)
  - gretelai/synthetic_text_to_sql (secondary, 100K synthetic, complexity-filtered)

Steps:
  load → concat → clean → format → stratified split
"""

import hashlib
import re
from typing import Optional

import pandas as pd
import sqlparse
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────

TRAIN_TEMPLATE = """\
### Task
Generate a SQL query to answer the following question.

### Database Schema
{context}

### Question
{question}

### SQL
{sql}"""

INFERENCE_TEMPLATE = """\
### Task
Generate a SQL query to answer the following question.

### Database Schema
{context}

### Question
{question}

### SQL
"""


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_sql_create_context() -> pd.DataFrame:
    """
    Load b-mc2/sql-create-context.
    Fields: question, context (CREATE TABLE ...), answer (SQL).
    All 78K rows used — this is the backbone of the training set.
    """
    print("  Loading b-mc2/sql-create-context ...")
    ds = load_dataset("b-mc2/sql-create-context", split="train")
    df = ds.to_pandas()[["question", "context", "answer"]]
    df = df.rename(columns={"answer": "sql"})
    df["source"] = "sql_create_context"
    print(f"  → {len(df):,} rows loaded")
    return df


def load_gretel_synthetic(max_samples: int = 15_000) -> pd.DataFrame:
    """
    Load gretelai/synthetic_text_to_sql — challenging + moderate queries only.
    Simple queries from Gretel add little value on top of sql-create-context.

    Fields: sql_prompt, sql_context, sql, sql_complexity.
    """
    print("  Loading gretelai/synthetic_text_to_sql (challenging + moderate) ...")
    ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
    df = ds.to_pandas()
    df = df[df["sql_complexity"].isin(["challenging", "moderate"])].copy()
    df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    df = df.rename(columns={"sql_prompt": "question", "sql_context": "context"})
    df = df[["question", "context", "sql"]]
    df["source"] = "gretel_synthetic"
    print(f"  → {len(df):,} rows selected")
    return df


# ─────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────

def _is_valid_sql(sql: str) -> bool:
    """Lightweight SQL parse check — rejects obviously broken strings."""
    try:
        stripped = sql.strip()
        if not stripped:
            return False
        parsed = sqlparse.parse(stripped)
        return len(parsed) > 0 and len(parsed[0].tokens) > 1
    except Exception:
        return False


def _is_select_only(sql: str) -> bool:
    """
    Reject destructive statements (INSERT/UPDATE/DELETE/DROP/CREATE/ALTER).
    Text-to-SQL for read queries should only produce SELECT statements.
    """
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    return first_word == "SELECT"


# ─────────────────────────────────────────────
# Complexity classifier
# ─────────────────────────────────────────────

def classify_complexity(sql: str) -> str:
    """
    Classify a SQL query into 4 complexity buckets.
    Used for stratified splitting and evaluation breakdown.

    Buckets (from easy to hard):
      easy        — simple SELECT/WHERE, single table
      medium      — GROUP BY / ORDER BY / HAVING / subquery
      hard        — single JOIN
      extra_hard  — multi-JOIN or JOIN + aggregation
    """
    s = sql.upper()
    join_count = s.count("JOIN")
    has_agg = any(k in s for k in ("GROUP BY", "HAVING"))
    has_subq = "(SELECT" in s.replace(" ", "")
    has_union = any(k in s for k in ("UNION", "INTERSECT", "EXCEPT"))

    if join_count >= 2 or (join_count >= 1 and (has_agg or has_subq)):
        return "extra_hard"
    if join_count == 1:
        return "hard"
    if has_agg or has_subq or has_union:
        return "medium"
    return "easy"


# ─────────────────────────────────────────────
# Cleaning pipeline
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning steps in order. Each step is logged.

    Steps:
      1. Drop rows with null/empty fields
      2. Validate SQL is parseable
      3. Keep SELECT-only queries
      4. Deduplicate by question hash (keep first occurrence, priority: sql_create_context > gretel)
      5. Add complexity label
    """
    n0 = len(df)

    # 1. Nulls and empties
    df = df.dropna(subset=["question", "context", "sql"])
    df = df[df["question"].str.strip().ne("")]
    df = df[df["context"].str.strip().ne("")]
    df = df[df["sql"].str.strip().ne("")]
    print(f"  After null/empty drop:   {len(df):,}  (removed {n0 - len(df):,})")
    n1 = len(df)

    # 2. Valid SQL
    df = df[df["sql"].apply(_is_valid_sql)].copy()
    print(f"  After SQL validation:    {len(df):,}  (removed {n1 - len(df):,})")
    n2 = len(df)

    # 3. SELECT-only (no DDL/DML leakage)
    df = df[df["sql"].apply(_is_select_only)].copy()
    print(f"  After SELECT-only filter:{len(df):,}  (removed {n2 - len(df):,})")
    n3 = len(df)

    # 4. Deduplication by question
    # Sort so sql_create_context rows are kept over gretel on conflict
    source_priority = {"sql_create_context": 0, "gretel_synthetic": 1}
    df["_priority"] = df["source"].map(source_priority)
    df = df.sort_values("_priority")
    df["_q_hash"] = df["question"].str.lower().str.strip().apply(
        lambda x: hashlib.md5(x.encode()).hexdigest()
    )
    df = df.drop_duplicates(subset=["_q_hash"], keep="first")
    df = df.drop(columns=["_priority", "_q_hash"])
    print(f"  After deduplication:     {len(df):,}  (removed {n3 - len(df):,})")

    # 5. Complexity
    df["complexity"] = df["sql"].apply(classify_complexity)
    df = df.reset_index(drop=True)

    print(f"\n  Final: {len(df):,} rows")
    print(f"  Complexity breakdown:\n{df['complexity'].value_counts().to_string()}")
    print(f"  Source breakdown:\n{df['source'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────

def format_for_training(row: pd.Series) -> str:
    return TRAIN_TEMPLATE.format(
        context=row["context"].strip(),
        question=row["question"].strip(),
        sql=row["sql"].strip(),
    )


def format_for_inference(question: str, context: str) -> str:
    return INFERENCE_TEMPLATE.format(
        context=context.strip(),
        question=question.strip(),
    )


# ─────────────────────────────────────────────
# Train / val / test split
# ─────────────────────────────────────────────

def stratified_split(
    df: pd.DataFrame,
    val_size: float = 0.08,
    test_size: float = 0.05,
    random_state: int = 42,
) -> DatasetDict:
    """
    Stratified split by complexity bucket.
    Ensures all complexity classes are represented in all splits.
    """
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["complexity"],
        random_state=random_state,
    )
    adjusted_val = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val,
        stratify=train_val["complexity"],
        random_state=random_state,
    )

    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name}: {len(split):,}  |  {split['complexity'].value_counts().to_dict()}")

    return DatasetDict({
        "train": Dataset.from_pandas(train.reset_index(drop=True)),
        "validation": Dataset.from_pandas(val.reset_index(drop=True)),
        "test": Dataset.from_pandas(test.reset_index(drop=True)),
    })


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def build_dataset(
    gretel_max_samples: int = 15_000,
    val_size: float = 0.08,
    test_size: float = 0.05,
    random_state: int = 42,
) -> DatasetDict:
    """
    Full pipeline: load → clean → format → split.

    Returns a DatasetDict with train / validation / test splits.
    Each example has columns: question, context, sql, source, complexity, text.
    The `text` column contains the fully formatted training prompt.
    """
    print("=" * 55)
    print("STEP 1: Loading datasets")
    print("=" * 55)
    df_sql = load_sql_create_context()
    df_gretel = load_gretel_synthetic(max_samples=gretel_max_samples)
    df = pd.concat([df_sql, df_gretel], ignore_index=True)
    print(f"\nCombined raw: {len(df):,} rows\n")

    print("=" * 55)
    print("STEP 2: Cleaning")
    print("=" * 55)
    df = clean(df)

    print("\n" + "=" * 55)
    print("STEP 3: Formatting prompts")
    print("=" * 55)
    df["text"] = df.apply(format_for_training, axis=1)
    avg_len = df["text"].str.len().mean()
    print(f"  Average prompt length: {avg_len:.0f} chars")

    print("\n" + "=" * 55)
    print("STEP 4: Stratified split")
    print("=" * 55)
    splits = stratified_split(df, val_size=val_size, test_size=test_size, random_state=random_state)

    return splits


# ─────────────────────────────────────────────
# SQL normalizer (used by evaluate.py)
# ─────────────────────────────────────────────

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL string for exact-match comparison.
    Strips whitespace, lowercases, formats keywords consistently.
    """
    sql = sql.strip()
    sql = re.sub(r"\s+", " ", sql)
    sql = sqlparse.format(sql, keyword_case="upper", strip_whitespace=True)
    return sql.strip().lower()
