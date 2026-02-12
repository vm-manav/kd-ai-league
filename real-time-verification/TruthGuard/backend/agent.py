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

# --- CORE LOGIC ---

def get_agent_response(claim: str):
    """
    Main function to verify a claim.
    Flow: Check DB -> If Miss, Search Web -> Verify -> Save Result
    """
    print(f"\n🔎 Analyzing Claim: '{claim}'")
    
    # 1. Check Knowledge Base (Vector DB)
    vector = embeddings.embed_query(claim)
    try:
        kb_results = index.query(vector=vector, top_k=1, include_metadata=True)
        kb_match = kb_results['matches'][0] if kb_results['matches'] else None
    except Exception as e:
        print(f"⚠️ Pinecone Error: {e}")
        kb_match = None

    # Logic: If we have a high confidence match (> 0.85), use it.
    kb_context = "No previous records found."
    if kb_match and kb_match['score'] > 0.85:
        data = kb_match['metadata']
        print(f"✅ Found in Knowledge Base (Score: {kb_match['score']:.2f})")
        return {
            "verdict": data['verdict'],
            "reasoning": data['reasoning'],
            "confidence_score": 100,
            "sources": [{"title": "TruthGuard Verified Archive", "url": "#", "id": 1}],
            "source_type": "ARCHIVE"
        }

    # 2. If no DB match, Search the Web
    print("🌐 Searching the Web...")
    try:
        web_results = search_tool.invoke({"query": claim})
    except Exception as e:
        return {"verdict": "Error", "reasoning": f"Search failed: {str(e)}", "sources": []}

    # Format web results for LLM
    web_context = "\n".join([f"Source [{i+1}]: {r['content']} (URL: {r['url']})" for i, r in enumerate(web_results)])

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
    response['sources'] = [{"id": i+1, "title": r['content'][:60]+"...", "url": r['url']} for i, r in enumerate(web_results)]
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