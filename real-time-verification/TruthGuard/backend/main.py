from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import get_agent_response
import uvicorn

app = FastAPI(title="TruthGuard Backend")

# Allow Chrome Extension to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to Extension ID
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VerifyRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "TruthGuard AI is Active"}

@app.post("/verify")
def verify_claim(request: VerifyRequest):
    claim = request.text.strip()
    if not claim:
        raise HTTPException(status_code=400, detail="No text provided")
    
    if len(claim) > 1000:
        raise HTTPException(status_code=400, detail="Text too long (max 1000 chars)")

    try:
        result = get_agent_response(claim)
        return result
    except Exception as e:
        print(f"SERVER ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)