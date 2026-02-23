# 🌍 Autonomous Travel Swarm

A Streamlit app powered by **LangGraph** with parallel agents (Booking, Weather, Local Expert) to plan trips. Get itineraries, weather alerts, and an interactive map with one prompt.

## Features

- **Parallel agents**: Booking, weather, and local-insider nodes run in a LangGraph workflow
- **Tavily search**: Real-time web search for flights, hotels, and local tips
- **Structured output**: Markdown itinerary + map markers (Folium) for key locations
- **HITL**: Human-in-the-loop approval step before “finalizing” bookings

## Setup

1. **Clone and enter the project**
   ```bash
   cd ai-travel-planning-agent
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment variables**  
   Create a `.env` file in this directory with:
   ```env
   OPENAI_API_KEY=your_openai_key
   TAVILY_API_KEY=your_tavily_key
   ```

## Run

```bash
source .venv/bin/activate
streamlit run app.py
```

Open the URL shown in the terminal (e.g. `http://localhost:8501`).

## Stack

- **Streamlit** – UI and chat
- **LangGraph** – Agent graph and checkpointing
- **LangChain** – LLM (OpenAI), tools (Tavily)
- **Folium / streamlit-folium** – Map and markers
- **Pydantic** – Structured outputs (trip intent, itinerary, map locations)
