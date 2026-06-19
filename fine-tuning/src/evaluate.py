"""
Evaluation suite for Text-to-SQL.

Metrics implemented:
  1. Exact Match (EM)         — normalized string match
  2. Execution Accuracy (EX)  — in-memory SQLite result comparison
  3. Complexity Breakdown     — EM + EX per complexity bucket
  4. Error Taxonomy           — classify failure modes
"""

import re
import sqlite3
import traceback
from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import sqlparse

from .data_pipeline import classify_complexity, normalize_sql


# ─────────────────────────────────────────────
# 1. Exact Match
# ─────────────────────────────────────────────

def exact_match(predicted: str, gold: str) -> bool:
    """Normalized exact match — case-insensitive, whitespace-collapsed."""
    return normalize_sql(predicted) == normalize_sql(gold)


def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """Corpus-level exact match accuracy."""
    assert len(predictions) == len(references)
    hits = sum(exact_match(p, r) for p, r in zip(predictions, references))
    return hits / len(predictions)


# ─────────────────────────────────────────────
# 2. Execution Accuracy
# ─────────────────────────────────────────────

def _build_db(schema_sql: str) -> Optional[sqlite3.Connection]:
    """
    Create an in-memory SQLite database from CREATE TABLE statements.
    Returns None if schema is malformed.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_sql)
        conn.commit()
        return conn
    except Exception:
        return None


def _run_query(conn: sqlite3.Connection, sql: str) -> Optional[set]:
    """
    Execute a SQL query and return result as a frozenset of row tuples.
    Returns None on any execution error (syntax, missing table, etc.).
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return frozenset(rows)
    except Exception:
        return None


def execution_accuracy_single(
    predicted_sql: str,
    gold_sql: str,
    schema_sql: str,
) -> dict[str, Any]:
    """
    Check execution accuracy for a single example.

    Returns:
        {
          "exec_match": bool,
          "predicted_error": str | None,
          "gold_error":      str | None,
          "db_error":        bool,
        }
    """
    result = {
        "exec_match": False,
        "predicted_error": None,
        "gold_error": None,
        "db_error": False,
    }

    conn = _build_db(schema_sql)
    if conn is None:
        result["db_error"] = True
        return result

    gold_rows = _run_query(conn, gold_sql)
    if gold_rows is None:
        result["gold_error"] = "gold_sql_failed"
        conn.close()
        return result

    pred_rows = _run_query(conn, predicted_sql)
    conn.close()

    if pred_rows is None:
        result["predicted_error"] = "predicted_sql_failed"
    else:
        result["exec_match"] = gold_rows == pred_rows

    return result


def execution_accuracy_score(
    predictions: list[str],
    references: list[str],
    schemas: list[str],
) -> dict[str, Any]:
    """
    Corpus-level execution accuracy.

    Returns dict with:
      - exec_accuracy:   float  (over examples where gold ran successfully)
      - db_errors:       int    (schema build failures — these rows are skipped)
      - gold_errors:     int    (gold SQL failed — also skipped)
      - pred_errors:     int    (predicted SQL failed — counted as incorrect)
      - n_evaluated:     int
    """
    assert len(predictions) == len(references) == len(schemas)

    total = len(predictions)
    exec_matches = 0
    db_errors = 0
    gold_errors = 0
    pred_errors = 0
    n_evaluated = 0

    for pred, gold, schema in zip(predictions, references, schemas):
        r = execution_accuracy_single(pred, gold, schema)
        if r["db_error"]:
            db_errors += 1
            continue
        if r["gold_error"]:
            gold_errors += 1
            continue
        n_evaluated += 1
        if r["exec_match"]:
            exec_matches += 1
        elif r["predicted_error"]:
            pred_errors += 1

    return {
        "exec_accuracy": exec_matches / n_evaluated if n_evaluated > 0 else 0.0,
        "exec_matches": exec_matches,
        "n_evaluated": n_evaluated,
        "db_errors": db_errors,
        "gold_errors": gold_errors,
        "pred_errors": pred_errors,
        "total": total,
    }


# ─────────────────────────────────────────────
# 3. Complexity Breakdown
# ─────────────────────────────────────────────

def complexity_breakdown(
    predictions: list[str],
    references: list[str],
    schemas: list[str],
    complexities: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Report EM and EX per complexity bucket.
    If complexities not provided, they are inferred from the gold SQL.
    """
    if complexities is None:
        complexities = [classify_complexity(r) for r in references]

    buckets: dict[str, dict] = defaultdict(lambda: {
        "em_hits": 0, "ex_hits": 0, "total": 0, "ex_skipped": 0
    })

    for pred, gold, schema, comp in zip(predictions, references, schemas, complexities):
        b = buckets[comp]
        b["total"] += 1
        if exact_match(pred, gold):
            b["em_hits"] += 1
        ex_r = execution_accuracy_single(pred, gold, schema)
        if ex_r["db_error"] or ex_r["gold_error"]:
            b["ex_skipped"] += 1
        elif ex_r["exec_match"]:
            b["ex_hits"] += 1

    rows = []
    order = ["easy", "medium", "hard", "extra_hard"]
    for comp in order:
        if comp not in buckets:
            continue
        b = buckets[comp]
        ex_denom = b["total"] - b["ex_skipped"]
        rows.append({
            "complexity": comp,
            "count": b["total"],
            "exact_match": round(b["em_hits"] / b["total"], 4) if b["total"] else 0.0,
            "exec_accuracy": round(b["ex_hits"] / ex_denom, 4) if ex_denom > 0 else None,
            "ex_skipped": b["ex_skipped"],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 4. Error Taxonomy
# ─────────────────────────────────────────────

_SYNTAX_ERROR_PATTERNS = [
    r"syntax error",
    r"near .+: syntax",
    r"incomplete input",
]


def _is_syntax_error(err_msg: str) -> bool:
    msg = err_msg.lower()
    return any(re.search(p, msg) for p in _SYNTAX_ERROR_PATTERNS)


def classify_error(predicted_sql: str, gold_sql: str, schema_sql: str) -> str:
    """
    Classify the type of error for a single failure case.

    Classes:
      exact_match        — normalized strings match (not an error)
      syntax_error       — predicted SQL fails to parse
      wrong_table        — predicted uses a table not in schema
      wrong_column       — predicted references a non-existent column alias
      wrong_aggregation  — aggregation function mismatch
      wrong_logic        — SQL runs but returns wrong rows
      execution_error    — predicted runs but errors out for unknown reason
      schema_skipped     — schema build failed, can't evaluate
    """
    if exact_match(predicted_sql, gold_sql):
        return "exact_match"

    conn = _build_db(schema_sql)
    if conn is None:
        return "schema_skipped"

    try:
        conn.execute(f"EXPLAIN {predicted_sql}")
    except sqlite3.OperationalError as e:
        msg = str(e).lower()
        conn.close()
        if _is_syntax_error(msg):
            return "syntax_error"
        if "no such table" in msg:
            return "wrong_table"
        if "no such column" in msg:
            return "wrong_column"
        return "execution_error"
    except Exception:
        conn.close()
        return "execution_error"

    pred_rows = _run_query(conn, predicted_sql)
    gold_rows = _run_query(conn, gold_sql)
    conn.close()

    if pred_rows is None:
        return "execution_error"
    if gold_rows is None:
        return "schema_skipped"

    pred_sql_u = predicted_sql.upper()
    gold_sql_u = gold_sql.upper()
    agg_fns = {"COUNT(", "SUM(", "AVG(", "MAX(", "MIN("}
    pred_aggs = {a for a in agg_fns if a in pred_sql_u}
    gold_aggs = {a for a in agg_fns if a in gold_sql_u}
    if pred_aggs != gold_aggs:
        return "wrong_aggregation"

    return "wrong_logic"


def error_taxonomy(
    predictions: list[str],
    references: list[str],
    schemas: list[str],
) -> pd.DataFrame:
    """
    Run error classification over all failed examples.
    Returns a DataFrame with counts and percentages per error class.
    """
    labels = []
    for pred, gold, schema in zip(predictions, references, schemas):
        labels.append(classify_error(pred, gold, schema))

    counts = pd.Series(labels).value_counts()
    df = pd.DataFrame({
        "error_type": counts.index,
        "count": counts.values,
        "pct": (counts.values / len(labels) * 100).round(2),
    })
    return df


# ─────────────────────────────────────────────
# Full evaluation report
# ─────────────────────────────────────────────

def full_report(
    predictions: list[str],
    references: list[str],
    schemas: list[str],
    complexities: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run all metrics and return a structured report dict.
    """
    em = exact_match_score(predictions, references)
    ex = execution_accuracy_score(predictions, references, schemas)
    breakdown = complexity_breakdown(predictions, references, schemas, complexities)
    taxonomy = error_taxonomy(predictions, references, schemas)

    report = {
        "exact_match": round(em, 4),
        "execution_accuracy": round(ex["exec_accuracy"], 4),
        "n_total": len(predictions),
        "n_evaluated_for_ex": ex["n_evaluated"],
        "breakdown_by_complexity": breakdown,
        "error_taxonomy": taxonomy,
    }

    if verbose:
        print("\n" + "=" * 55)
        print("EVALUATION REPORT")
        print("=" * 55)
        print(f"  Exact Match (EM):       {em:.2%}")
        print(f"  Execution Accuracy (EX):{ex['exec_accuracy']:.2%}  "
              f"(evaluated on {ex['n_evaluated']}/{len(predictions)} examples)")
        print(f"  DB build errors:        {ex['db_errors']}")
        print(f"  Gold SQL errors:        {ex['gold_errors']}")
        print(f"  Predicted SQL errors:   {ex['pred_errors']}")
        print("\n  Complexity Breakdown:")
        print(breakdown.to_string(index=False))
        print("\n  Error Taxonomy:")
        print(taxonomy.to_string(index=False))

    return report
