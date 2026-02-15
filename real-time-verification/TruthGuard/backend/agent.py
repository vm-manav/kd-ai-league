import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
LLM_MODEL = "gpt-4o"  # Use gpt-4-turbo or gpt-3.5-turbo if 4o is not available
EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_SCORE_THRESHOLD = 0.85   # Use cache only if exactly one match above this
MIN_EVIDENCE_CHARS = 100          # Below this total content length → "Not enough evidence"

# --- INITIALIZATION ---
print("🔌 Initializing TruthGuard Agent...")

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
search_tool = TavilySearchResults(max_results=5)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Automatic Index Creation (Hackathon Helper)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"⚠️ Index '{INDEX_NAME}' not found. Creating it now...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536, # Matches text-embedding-3-small
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    time.sleep(10) # Wait for index to be ready

index = pc.Index(INDEX_NAME)

# --- HELPERS ---

def classify_claim(claim: str) -> str:
    """
    Classify claim into: yes (verifiable), no_opinion (subjective), no_vague (lacks context).
    Returns one of: "yes", "no_opinion", "no_vague".
    """
    classifier_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a classifier. Answer with exactly one of: YES, NO_OPINION, NO_VAGUE.

- YES: The input is a specific factual claim that can be verified as true or false (who/what/when/where are clear enough).
- NO_OPINION: The input is subjective, a value judgment, or a preference (e.g. "best", "worst", "should", "I think"). Not fact-checkable.
- NO_VAGUE: The input is factual in principle but lacks sufficient context to verify (e.g. "the government did X" without which government or when)."""),
        ("human", "Classify this input:\n\n{claim}"),
    ])
    chain = classifier_prompt | llm
    out = chain.invoke({"claim": claim}).content.strip().upper()
    if "NO_VAGUE" in out or "VAGUE" in out:
        return "no_vague"
    if "NO_OPINION" in out or "OPINION" in out:
        return "no_opinion"
    if out.startswith("YES"):
        return "yes"
    # Default: treat as not verifiable
    return "no_opinion"

def _not_factual_response():
    return {
        "verdict": "Not a factual claim",
        "reasoning": "This is not a verifiable factual claim.",
        "confidence_score": 0,
        "sources": [],
        "source_type": "CLASSIFIER",
    }

def _insufficient_context_response():
    return {
        "verdict": "Unverified",
        "reasoning": "Claim lacks sufficient context.",
        "confidence_score": 0,
        "sources": [],
        "source_type": "CLASSIFIER",
    }

def _not_enough_evidence_response():
    return {
        "verdict": "Unverified",
        "reasoning": "Not enough evidence to verify this claim.",
        "confidence_score": 0,
        "sources": [],
        "source_type": "INSUFFICIENT_EVIDENCE",
    }

# --- CORE LOGIC ---

def get_agent_response(claim: str):
    """
    Main function to verify a claim.
    Flow: Classifier -> Pinecone (top 3, Option B) -> Tavily -> Not enough evidence? -> LLM -> Save if high confidence
    """
    print(f"\n🔎 Analyzing Claim: '{claim}'")

    # 0. Claim classifier: verifiable, opinion, or vague?
    print("📋 Checking if input is a verifiable factual claim...")
    classification = classify_claim(claim)
    if classification == "no_opinion":
        print("⏭️ Not a factual claim (opinion/subjective) — skipping verification.")
        return _not_factual_response()
    if classification == "no_vague":
        print("⏭️ Claim lacks sufficient context — skipping verification.")
        return _insufficient_context_response()

    # 1. Check Knowledge Base (Vector DB) — top_k=3, use cache only if exactly one above threshold (Option B)
    vector = embeddings.embed_query(claim)
    try:
        kb_results = index.query(vector=vector, top_k=3, include_metadata=True)
        matches = kb_results.get("matches") or []
    except Exception as e:
        print(f"⚠️ Pinecone Error: {e}")
        matches = []

    strong_matches = [m for m in matches if m.get("score") and m["score"] > PINECONE_SCORE_THRESHOLD]
    if len(strong_matches) == 1:
        kb_match = strong_matches[0]
        data = kb_match.get("metadata") or {}
        print(f"✅ Found in Knowledge Base (Score: {kb_match['score']:.2f})")
        return {
            "verdict": data.get("verdict", "Unverified"),
            "reasoning": data.get("reasoning", "Previously verified."),
            "confidence_score": 100,
            "sources": [{"title": "TruthGuard Verified Archive", "url": "#", "id": 1}],
            "source_type": "ARCHIVE",
        }

    # 2. Search the Web
    print("🌐 Searching the Web...")
    try:
        web_results = search_tool.invoke({"query": claim})
    except Exception as e:
        return {"verdict": "Error", "reasoning": f"Search failed: {str(e)}", "sources": []}

    # 2b. Not enough evidence? Avoid forcing LLM to guess.
    total_content = sum(len((r or {}).get("content", "")) for r in (web_results or []))
    if not web_results or total_content < MIN_EVIDENCE_CHARS:
        print("⚠️ Insufficient web evidence — returning Unverified.")
        return _not_enough_evidence_response()

    # Format web results for LLM
    web_context = "\n".join([
        f"Source [{i+1}]: {(r or {}).get('content', '')} (URL: {(r or {}).get('url', '')})"
        for i, r in enumerate(web_results)
    ])

    # 3. The Verdict (LLM Analysis)
    print("⚖️ Verifying Evidence...")
    parser = JsonOutputParser()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are TruthGuard, an AI Fact-Checking Agent.
        
        INPUT DATA:
        User Claim: {claim}
        Web Evidence: {web_context}
        
        TASK:
        1. Analyze if the evidence supports or refutes the claim.
        2. Assign a verdict: "True", "False", "Misleading", or "Unverified".
        3. Explain your reasoning in 2 short sentences.
        4. Give a confidence score (0-100).
        
        OUTPUT JSON FORMAT:
        {{
            "verdict": "True/False/Misleading/Unverified",
            "reasoning": "string",
            "confidence_score": int
        }}
        """),
    ])

    chain = prompt | llm | parser
    response = chain.invoke({"claim": claim, "web_context": web_context})

    # Add sources back to response
    response['sources'] = [
        {"id": i+1, "title": (r.get("content") or "")[:60] + "...", "url": r.get("url", "")}
        for i, r in enumerate(web_results)
    ]
    response['source_type'] = "LIVE_WEB"

    # 4. Learning Loop (Save to DB)
    if response['confidence_score'] > 85:
        print("💾 Saving Verified Fact to DB...")
        meta = {
            "text": claim,
            "verdict": response['verdict'],
            "reasoning": response['reasoning']
        }
        # Upsert to Pinecone
        index.upsert(vectors=[(str(hash(claim)), vector, meta)])

    return response