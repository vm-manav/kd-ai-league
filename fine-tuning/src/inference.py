"""
Inference pipeline for the fine-tuned Text-to-SQL model.

Covers:
  - Local generation (PEFT adapters over quantized base)
  - Batch inference for evaluation
  - SQL extraction from raw generation output
  - FastAPI serving stub (production thinking, Step 7)
"""

import re
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .data_pipeline import format_for_inference

# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────

def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    """
    Load the base model (optionally 4-bit quantized) and LoRA adapters.

    Args:
        base_model_name: HuggingFace model ID or local path.
        adapter_path:    Path to saved LoRA adapter weights. None = base model only.
        load_in_4bit:    Whether to load in 4-bit NF4 (QLoRA inference).
        device_map:      "auto" lets HF handle GPU/CPU placement.

    Returns:
        (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────
# SQL extractor
# ─────────────────────────────────────────────

def extract_sql(raw_output: str) -> str:
    """
    Extract the SQL portion from model generation output.

    The model is trained with the prompt ending in '### SQL\n',
    so we strip the prompt prefix and take everything up to
    the next '###' marker or end of string.
    """
    # If the model repeated the prompt, extract after the last ### SQL
    marker = "### SQL"
    if marker in raw_output:
        sql = raw_output.split(marker)[-1].strip()
    else:
        sql = raw_output.strip()

    # Stop at the next section marker if present
    sql = re.split(r"\n###", sql)[0].strip()

    # Strip code fences if model wraps in markdown
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql).strip()

    return sql


# ─────────────────────────────────────────────
# Single inference
# ─────────────────────────────────────────────

def generate_sql(
    model,
    tokenizer,
    question: str,
    context: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    do_sample: bool = False,
) -> str:
    """
    Generate SQL for a single (question, schema) pair.

    temperature=0.0 + do_sample=False → greedy decoding.
    Use temperature>0 + do_sample=True for diversity (not recommended for eval).
    """
    prompt = format_for_inference(question, context)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return extract_sql(generated)


# ─────────────────────────────────────────────
# Batch inference (for evaluation)
# ─────────────────────────────────────────────

def batch_generate_sql(
    model,
    tokenizer,
    questions: list[str],
    contexts: list[str],
    batch_size: int = 4,
    max_new_tokens: int = 200,
) -> list[str]:
    """
    Run batch inference over a list of (question, context) pairs.
    Returns list of extracted SQL strings.
    """
    assert len(questions) == len(contexts)
    results = []

    for i in range(0, len(questions), batch_size):
        batch_q = questions[i : i + batch_size]
        batch_c = contexts[i : i + batch_size]
        prompts = [format_for_inference(q, c) for q, c in zip(batch_q, batch_c)]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            results.append(extract_sql(generated))

        if (i // batch_size) % 10 == 0:
            print(f"  Batch {i // batch_size + 1}/{(len(questions) + batch_size - 1) // batch_size}")

    return results


# ─────────────────────────────────────────────
# Step 7: Production serving stub (FastAPI)
# ─────────────────────────────────────────────

FASTAPI_STUB = '''
"""
Production serving example — FastAPI endpoint.

Deploy with:
    uvicorn serve:app --host 0.0.0.0 --port 8000

Or containerise with the provided Dockerfile.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from inference import load_model_and_tokenizer, generate_sql

app = FastAPI(title="Text-to-SQL API", version="1.0.0")

# Load once at startup
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "./outputs/qlora_run/final_adapter"

model, tokenizer = load_model_and_tokenizer(
    base_model_name=MODEL_NAME,
    adapter_path=ADAPTER_PATH,
    load_in_4bit=True,
)


class QueryRequest(BaseModel):
    question: str           # Natural language question
    schema: str             # CREATE TABLE statements for the target DB
    max_new_tokens: int = 200


class QueryResponse(BaseModel):
    sql: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/generate", response_model=QueryResponse)
def generate(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    if not req.schema.strip():
        raise HTTPException(status_code=400, detail="schema cannot be empty")

    sql = generate_sql(
        model=model,
        tokenizer=tokenizer,
        question=req.question,
        context=req.schema,
        max_new_tokens=req.max_new_tokens,
    )
    return QueryResponse(sql=sql, question=req.question)
'''


def write_serve_file(path: str = "serve.py") -> None:
    """Write the FastAPI serving stub to disk."""
    with open(path, "w") as f:
        f.write(FASTAPI_STUB.strip())
    print(f"FastAPI serving stub written to {path}")
