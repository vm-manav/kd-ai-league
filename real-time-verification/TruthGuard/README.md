# TruthGuard — AI Fact Checker

Browser extension + backend for real-time claim verification using Agentic RAG (LangChain, OpenAI, Tavily, Pinecone).

## Structure

- **backend/** — FastAPI server and verification agent
- **extension/** — Chrome extension (popup + content script)

## Setup

### Backend

1. Go to the backend folder and create a virtual environment:

   ```bash
   cd real-time-verification/TruthGuard/backend
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:

   - `OPENAI_API_KEY` — OpenAI API key
   - `TAVILY_API_KEY` — Tavily search API key
   - `PINECONE_API_KEY` — Pinecone API key
   - `PINECONE_INDEX_NAME` — Pinecone index name used by the agent

4. Run the server:

   ```bash
   python main.py
   ```

   API runs at **http://localhost:8000**.

### Extension

1. Open Chrome and go to **chrome://extensions**
2. Turn on **Developer mode**
3. Click **Load unpacked** and select the `extension` folder inside this directory

## Usage

- **Highlight text** on any webpage — it appears under “Selected claim” in the popup.
- **Type or paste** a claim in the “Or type / paste claim” box.
- Click **Verify claim** to get a verdict (True / False / Misleading), reasoning, and sources.

The extension talks to the backend at `http://localhost:8000`, so keep the backend running while using the extension.

## Security

- Never commit `.env` or any file containing API keys.
- Use `.env.example` as a template; the real `.env` is listed in `.gitignore`.
