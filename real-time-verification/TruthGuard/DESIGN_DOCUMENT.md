# TruthGuard — Real-Time Claim Verification  
## Design Document

**Version:** 1.0  
**Scope:** real-time-verification project, TruthGuard backend and browser extension  
**Last updated:** February 2025  

---

## 1. Executive Summary

**TruthGuard** is an AI-powered fact-checking system that verifies textual claims in real time. It combines a **Chrome extension** (for capturing claims from the web) with a **Python backend** that uses an **Agentic RAG** pipeline: claim classification, vector search over a verified-knowledge base (Pinecone), live web search (Tavily), and an LLM (OpenAI) to produce a verdict. High-confidence results are stored back into the knowledge base for future cache hits, forming a learning loop.

---

## 2. Project Structure

```
real-time-verification/
└── TruthGuard/
    ├── backend/                    # FastAPI server + verification agent
    │   ├── main.py                 # API entrypoints, CORS, request validation
    │   ├── agent.py                # Core verification logic (classifier, RAG, LLM, learning)
    │   ├── requirements.txt        # Python dependencies
    │   ├── .env.example            # Template for API keys
    │   └── .env                    # Actual keys (gitignored)
    └── extension/                  # Chrome extension (Manifest V3)
        ├── manifest.json           # Permissions, content/background scripts, popup
        ├── popup.html              # Popup UI structure
        ├── popup.js                # Popup logic: selection, input, API call, result display
        ├── styles.css              # Popup styling and verdict colors
        ├── content.js              # Injected script: returns page selection to popup
        ├── background.js           # Service worker: context menu "Verify with TruthGuard"
        └── icon.png                # Extension icon
```

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER (Browser)                                   │
│  • Highlights text on any webpage  OR  types/pastes claim in popup           │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CHROME EXTENSION (TruthGuard)                              │
│  • content.js: gets selected text from page                                  │
│  • popup.js: shows selection + textarea, calls backend on "Verify claim"      │
│  • background.js: context menu "Verify with TruthGuard AI" (optional UX)     │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │ HTTP POST /verify { "text": "<claim>" }
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKEND (FastAPI @ localhost:8000)                         │
│  • main.py: validates request (non-empty, max 1000 chars), CORS              │
│  • Calls agent.get_agent_response(claim)                                      │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AGENT (agent.py) — Agentic RAG Pipeline                    │
│  0. Classifier (LLM)     → Is claim verifiable? (yes / no_opinion / no_vague)│
│  1. Pinecone (vector DB) → Top 3 matches; cache hit if exactly 1 > 0.85      │
│  2. Tavily               → Web search (up to 5 results)                      │
│  2b. Evidence check      → If too little content → return "Unverified"       │
│  3. LLM                  → Verdict (True/False/Misleading/Unverified)       │
│  4. Learning             → If confidence > 85, upsert claim+verdict to Pinecone│
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RESPONSE TO EXTENSION                                      │
│  { verdict, reasoning, confidence_score, sources[], source_type }            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. End-to-End User Flow

| Step | Actor | Action |
|------|--------|--------|
| 1 | User | Opens any webpage, optionally highlights a claim, or ignores selection. |
| 2 | User | Clicks TruthGuard extension icon → popup opens. |
| 3 | Extension | **content.js** runs in active tab; popup requests selection via `chrome.tabs.sendMessage(..., { action: "getSelection" })`. |
| 4 | content.js | Responds with `{ selection: window.getSelection().toString().trim() }`. Popup shows this under "Selected claim" or "No text selected on page." |
| 5 | User | Either keeps selected claim or types/pastes in "Or type / paste claim". Clicks **Verify claim**. |
| 6 | popup.js | Uses typed text if non-empty, else selected text. Disables button, shows loader "Scanning Knowledge Base & Web...". |
| 7 | popup.js | `POST http://localhost:8000/verify` with `{ text: "<claim>" }`. |
| 8 | Backend | Validates (length, non-empty), calls `get_agent_response(claim)`. |
| 9 | Agent | Runs full pipeline (see Section 5). Returns JSON. |
| 10 | Backend | Returns agent JSON to extension. |
| 11 | popup.js | Hides loader, calls `displayResult(data)`: verdict box (color by verdict), reasoning, list of sources. |
| 12 | User | Reads verdict (True / False / Misleading / Unverified / Not a factual claim), reasoning, and can open source links. |

---

## 5. Agent Pipeline (agent.py) — Detailed Flow

This section describes every step inside `get_agent_response(claim)`.

---

### 5.1 Configuration and Initialization

| Constant / Config | Purpose |
|-------------------|---------|
| `LLM_MODEL` | OpenAI model for classification and verdict (e.g. `gpt-4o`). |
| `EMBEDDING_MODEL` | Model for embedding the claim (e.g. `text-embedding-3-small`). |
| `INDEX_NAME` | Pinecone index name (from env). |
| `PINECONE_SCORE_THRESHOLD` (0.85) | Cache is used only when **exactly one** match has score above this. |
| `MIN_EVIDENCE_CHARS` (100) | Total character count of web snippets below this → return "Unverified" without asking LLM. |

**Initialization (on import):**

- Load `.env` via `load_dotenv()`.
- Create `ChatOpenAI` (temperature=0) and `OpenAIEmbeddings`.
- Create Tavily search tool: `TavilySearchResults(max_results=5)`.
- Connect to Pinecone; if index does not exist, create it (dimension=1536, cosine, serverless AWS us-east-1) and wait 10 seconds.
- Obtain `index = pc.Index(INDEX_NAME)` for queries and upserts.

---

### 5.2 Step 0 — Claim Classification

**Purpose:** Avoid fact-checking non-factual or under-specified inputs.

**Function:** `classify_claim(claim: str) -> str`

**Process:**

1. A **system + user** prompt is built with `ChatPromptTemplate`. The system instructs the LLM to answer with exactly one of: **YES**, **NO_OPINION**, **NO_VAGUE**.
   - **YES:** Specific factual claim that can be verified (who/what/when/where clear enough).
   - **NO_OPINION:** Subjective, value judgment, or preference (e.g. "best", "should", "I think") — not fact-checkable.
   - **NO_VAGUE:** Factual in principle but insufficient context (e.g. "the government did X" without which government or when).
2. The chain `classifier_prompt | llm` is invoked with `{ claim }`.
3. The raw output is normalized (strip, upper) and parsed:
   - If output contains "NO_VAGUE" or "VAGUE" → return `"no_vague"`.
   - Else if "NO_OPINION" or "OPINION" → return `"no_opinion"`.
   - Else if output starts with "YES" → return `"yes"`.
   - Default → `"no_opinion"` (treat as not verifiable).

**Early exits in `get_agent_response`:**

- If `classification == "no_opinion"` → return `_not_factual_response()`: verdict "Not a factual claim", reasoning, confidence 0, empty sources, `source_type: "CLASSIFIER"`.
- If `classification == "no_vague"` → return `_insufficient_context_response()`: verdict "Unverified", reasoning "Claim lacks sufficient context.", confidence 0, empty sources, `source_type: "CLASSIFIER"`.

Only when `classification == "yes"` does the pipeline continue to the knowledge base.

---

### 5.3 Step 1 — Knowledge Base (Pinecone) Lookup

**Purpose:** Reuse previous high-confidence verifications and avoid redundant web search + LLM calls.

**Process:**

1. **Embed the claim:** `vector = embeddings.embed_query(claim)` (same dimension as index, e.g. 1536).
2. **Query Pinecone:** `index.query(vector=vector, top_k=3, include_metadata=True)`. On exception, `matches = []`.
3. **Filter strong matches:** `strong_matches = [m for m in matches if m.get("score") and m["score"] > PINECONE_SCORE_THRESHOLD]`.
4. **Cache hit condition (Option B):** If **exactly one** strong match:
   - Take that match’s `metadata` (e.g. `verdict`, `reasoning`).
   - Return immediately with: `verdict`, `reasoning`, `confidence_score: 100`, `sources: [{ title: "TruthGuard Verified Archive", url: "#", id: 1 }]`, `source_type: "ARCHIVE"`.
5. If zero or more than one strong match, do **not** use cache; proceed to web search.

---

### 5.4 Step 2 — Web Search (Tavily)

**Purpose:** Retrieve current, external evidence for the claim.

**Process:**

1. Invoke `search_tool.invoke({"query": claim})` (Tavily, up to 5 results). On exception, return an error response (verdict "Error", reasoning with message, empty sources).
2. **Evidence sufficiency (Step 2b):**  
   `total_content = sum(len((r or {}).get("content", "")) for r in (web_results or []))`.  
   If `not web_results` or `total_content < MIN_EVIDENCE_CHARS` (100):
   - Return `_not_enough_evidence_response()`: verdict "Unverified", reasoning "Not enough evidence to verify this claim.", confidence 0, empty sources, `source_type: "INSUFFICIENT_EVIDENCE"`.
3. **Format for LLM:** Build `web_context` as a single string: for each result, a line like `Source [i]: <content> (URL: <url>)`.

---

### 5.5 Step 3 — LLM Verdict

**Purpose:** Analyze web evidence and produce a structured verdict.

**Process:**

1. **Prompt:** System message defines TruthGuard’s role. Placeholders: `{claim}`, `{web_context}`. Instructions:
   - Analyze whether evidence supports or refutes the claim.
   - Assign verdict: **"True"**, **"False"**, **"Misleading"**, or **"Unverified"**.
   - Explain in 2 short sentences.
   - Give confidence score (0–100).
   - Output must be JSON: `{ "verdict", "reasoning", "confidence_score" }`.
2. **Chain:** `prompt | llm | JsonOutputParser`. Invoke with `claim` and `web_context`.
3. **Enrich response:** Add `sources` (from Tavily results: id, title as truncated content, url) and `source_type: "LIVE_WEB"`.

---

### 5.6 Step 4 — Learning Loop (Save to Knowledge Base)

**Purpose:** Grow the verified archive so future similar claims can be answered from cache.

**Condition:** `response['confidence_score'] > 85`.

**Process:**

1. Build metadata: `text` (original claim), `verdict`, `reasoning`.
2. **Upsert to Pinecone:** `index.upsert(vectors=[(str(hash(claim)), vector, meta)])`.  
   - ID: `str(hash(claim))` (deterministic per claim text).  
   - Vector: same `vector` used for the query in Step 1.  
   - Metadata: `meta`.

After this, the function returns the same `response` (verdict, reasoning, confidence_score, sources, source_type) to the backend, which forwards it to the extension.

---

## 6. Data Models and API Contract

### 6.1 Request (Extension → Backend)

- **Endpoint:** `POST /verify`
- **Headers:** `Content-Type: application/json`
- **Body:** `{ "text": "<claim string>" }`
- **Validation (main.py):**
  - `text` must be non-empty after strip.
  - Length ≤ 1000 characters; otherwise 400 with message "Text too long (max 1000 chars)".

### 6.2 Response (Backend → Extension)

JSON object with:

| Field | Type | Description |
|-------|------|-------------|
| `verdict` | string | One of: "True", "False", "Misleading", "Unverified", "Not a factual claim", or "Error". |
| `reasoning` | string | Short explanation (2 sentences from LLM or fixed message for classifier/insufficient evidence). |
| `confidence_score` | number | 0–100; 0 for classifier/insufficient evidence, 100 for archive hit. |
| `sources` | array | List of `{ id?, title, url }` (optional id for display order). |
| `source_type` | string | "CLASSIFIER" \| "ARCHIVE" \| "INSUFFICIENT_EVIDENCE" \| "LIVE_WEB". |

### 6.3 Pinecone Stored Metadata (per vector)

- `text`: Original claim.
- `verdict`: Verdict string from LLM.
- `reasoning`: Reasoning string from LLM.

Vector ID: `str(hash(claim))` (Python `hash` of the claim string).

---

## 7. Extension Components in Detail

### 7.1 manifest.json

- **Manifest version:** 3.
- **Permissions:** `contextMenus`, `activeTab`, `scripting`.
- **Host permissions:** `http://localhost:8000/*` (for backend API).
- **Content script:** Injected into `<all_urls>`, script `content.js` — used to read selection.
- **Background:** Service worker `background.js` — creates context menu "Verify with TruthGuard AI" on selection.
- **Action:** Popup `popup.html`, icon `icon.png`.

### 7.2 content.js

- Listens for `chrome.runtime.onMessage` with `request.action === "getSelection"`.
- Replies with `sendResponse({ selection: window.getSelection().toString().trim() })`.

### 7.3 popup.js

- **On load:** Query active tab, send `getSelection` message, show result in `#claim-text`; if no selection, show "No text selected on page." Enable/disable Verify button based on typed or selected text.
- **Verify button:** Prefer `#claim-input` value if non-empty, else selected text. POST to `http://localhost:8000/verify`, then call `displayResult(data)`.
- **displayResult:** Shows verdict in `#verdict-box` with CSS class by verdict type (v-true, v-false, v-misleading, v-unverified, v-not-factual); fills reasoning and source list (links with title and url).

### 7.4 background.js

- On install, creates context menu item "Verify with TruthGuard AI" for context `selection`.
- On click: runs a small script in the tab that alerts the user to open the TruthGuard popup (MV3 limits programmatic popup opening).

### 7.5 Styling (styles.css)

- Popup width 350px; header dark; claim sections with labels and textarea; primary button; hidden loader (spinner + text); result area with verdict box, reasoning, and source list.
- Verdict colors: green (true), red (false), orange (misleading), grey (unverified), indigo (not factual).

---

## 8. Error Handling Summary

| Location | Situation | Behavior |
|----------|-----------|----------|
| main.py | Empty or missing text | 400, "No text provided". |
| main.py | Text length > 1000 | 400, "Text too long (max 1000 chars)". |
| main.py | Any exception in `get_agent_response` | 500, detail=str(e). |
| agent.py | Pinecone query fails | Treat as no matches; continue to Tavily. |
| agent.py | Tavily search fails | Return verdict "Error", reasoning with exception message. |
| agent.py | Too little web content | Return Unverified, source_type INSUFFICIENT_EVIDENCE. |
| popup.js | response.ok false or network error | Alert user to ensure backend is running at localhost:8000. |

---

## 9. Security and Deployment Notes

- **API keys:** Stored in backend `.env` (OPENAI_API_KEY, TAVILY_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME). Never committed; `.env.example` is the template.
- **CORS:** Backend allows all origins (`allow_origins=["*"]`). For production, restrict to the extension’s origin or ID.
- **Backend reachability:** Extension is hardcoded to `http://localhost:8000`. For production, use a configurable base URL and HTTPS.
- **Pinecone:** Index created automatically if missing (dimension 1536, cosine, serverless AWS). In production, consider pre-provisioning and stricter access control.

---

## 10. Summary Table — What Happens Where

| Step | Where | What |
|------|--------|------|
| User input | Extension (popup) | Selection from page and/or typed/pasted claim. |
| Send claim | popup.js | POST /verify with JSON body. |
| Validate | main.py | Non-empty, length ≤ 1000. |
| Classify | agent.py | LLM: YES / NO_OPINION / NO_VAGUE; early return if not YES. |
| Cache lookup | agent.py | Embed claim → Pinecone top_k=3 → if exactly 1 score > 0.85 → return archive response. |
| Web search | agent.py | Tavily 5 results; if total content < 100 chars → Unverified. |
| Verdict | agent.py | LLM + JSON parser → verdict, reasoning, confidence; add sources. |
| Learning | agent.py | If confidence > 85 → upsert (id=hash(claim), vector, metadata) to Pinecone. |
| Display | popup.js | Verdict color, reasoning text, source links. |

This design document reflects the full flow and responsibilities of the real-time-verification TruthGuard project and can be copied into a Google Doc for sharing or further editing.
