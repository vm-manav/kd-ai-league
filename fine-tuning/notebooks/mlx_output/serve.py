from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re
from mlx_lm import load, generate

MODEL_NAME   = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
ADAPTER_PATH = "./mlx_output/best_adapter"

model, tokenizer = load(MODEL_NAME, adapter_path=ADAPTER_PATH)
app = FastAPI(title="Text-to-SQL (MLX)", version="1.0.0")

TEMPLATE = (
    "### Task\n"
    "Generate a SQL query to answer the following question.\n\n"
    "### Database Schema\n{schema}\n\n"
    "### Question\n{question}\n\n"
    "### SQL\n"
)

BLOCKED_KW = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "EXEC"]
PII_FIELDS = ["password", "passwd", "credit_card", "ssn"]

class Req(BaseModel):
    question: str
    schema: str
    max_tokens: int = 200

def validate_sql(sql: str, schema: str):
    sql_u = sql.strip().upper()
    if not sql_u.startswith("SELECT"):
        return False, "Non-SELECT blocked"
    for kw in BLOCKED_KW:
        if re.search(rf"\\b{kw}\\b", sql_u):
            return False, f"Blocked keyword: {kw}"
    for pii in PII_FIELDS:
        if pii in sql.lower() and pii not in schema.lower():
            return False, f"PII field not in schema: {pii}"
    return True, "ok"

@app.get("/health")
def health(): return {"status": "ok", "model": MODEL_NAME}

@app.post("/generate")
def gen(req: Req):
    if not req.question.strip(): raise HTTPException(400, "question required")
    if not req.schema.strip():   raise HTTPException(400, "schema required")
    prompt = TEMPLATE.format(schema=req.schema, question=req.question)
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=req.max_tokens, verbose=False)
    sql = raw.split("### SQL")[-1].strip() if "### SQL" in raw else raw.strip()
    sql = re.split(r"\n###|\n```", sql)[0].strip()
    ok, reason = validate_sql(sql, req.schema)
    if not ok:
        raise HTTPException(422, f"Blocked: {reason}")
    return {"sql": sql, "question": req.question}