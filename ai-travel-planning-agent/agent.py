import os
import json
from typing import TypedDict, Annotated, List, Optional, Literal
import operator
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# --- PYDANTIC MODELS ---
class TripIntent(BaseModel):
    destination: str = Field(description="The city to travel to.")
    budget: Optional[int] = Field(description="Total budget in INR. Return None if not specified.")
    travel_style: Literal["Luxury", "Backpacker", "Family", "Adventure", "Standard"] = Field(...)

class MapLocation(BaseModel):
    name: str = Field(description="Name of the hotel or activity")
    lat: float = Field(description="Latitude")
    lon: float = Field(description="Longitude")

class ItineraryOutput(BaseModel):
    markdown_text: str = Field(description="The full formatted markdown itinerary")
    map_markers: List[MapLocation] = Field(description="List of 3-5 key locations for the map")

# --- STATE MEMORY ---
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_request: str
    destination: str
    budget: Optional[int]
    travel_style: str
    flights_data: str
    hotels_data: str
    insider_tips: str
    weather_alert: str       
    final_itinerary: str
    map_data: List[dict]     
    missing_info_question: str
    summary: str             

# --- AGENT NODES ---
def context_compactor(state: TravelState):
    """Summarizes long conversations to save API tokens and prevent hallucinations."""
    messages = state.get("messages", [])
    if len(messages) > 6:
        print("🗜️ Context getting long. Compacting memory...")
        summary_prompt = "Summarize the user's core travel preferences from this conversation so far."
        summary = llm.invoke([SystemMessage(content=summary_prompt)] + messages)
        return {"summary": summary.content}
    return {}

def intent_analyzer(state: TravelState):
    print("🧠 Analyzing Intent & Checking for Missing Info...")
    structured_llm = llm.with_structured_output(TripIntent)
    
    context = state.get("summary", "") + "\nLatest request: " + state['user_request']
    intent = structured_llm.invoke(context)
    
    question = ""
    if intent.budget is None:
        question = f"I'd love to plan your trip to {intent.destination}! What is your approximate budget in INR?"
        
    return {
        "destination": intent.destination, 
        "budget": intent.budget,
        "travel_style": intent.travel_style,
        "missing_info_question": question
    }

def fetch_bookings(state: TravelState):
    print("✈️ [Worker 1] Fetching Flights & Hotels...")
    simulated_hotels = f"Hotel: Taj {state.get('destination')}. Price: ₹8,000/night."
    simulated_flights = f"Flight: Vistara to {state.get('destination')}. Price: ₹6,500."
    return {"flights_data": simulated_flights, "hotels_data": simulated_hotels}

def fetch_tips(state: TravelState):
    print("🕵️ [Worker 2] Scouring Reddit for secrets...")
    search = TavilySearchResults(max_results=2)
    query = f"site:reddit.com best {state['travel_style']} food in {state['destination']}"
    results = search.invoke({"query": query})
    tips = "\n".join([f"- {r['content']}" for r in results])
    return {"insider_tips": tips}

def check_weather(state: TravelState):
    print("🌤️ [Worker 3] Checking Weather & Disasters...")
    dest = state.get('destination', '').lower()
    if "kerala" in dest or "mumbai" in dest:
        alert = "⚠️ WARNING: Heavy monsoon rains expected. Added indoor backup activities to itinerary."
    elif "rishikesh" in dest:
        alert = "✅ Weather is clear. Perfect conditions for river rafting."
    else:
        alert = "✅ Weather is pleasant. Standard packing recommended."
    return {"weather_alert": alert}

def compile_itinerary(state: TravelState):
    print("📝 Compiling Final Itinerary & Map Coordinates...")
    sys_prompt = f"""You are an elite Travel AI. Create a {state['travel_style']} itinerary.
    Budget: ₹{state['budget']}
    Flights: {state['flights_data']}
    Hotels: {state['hotels_data']}
    Reddit Tips: {state['insider_tips']}
    Weather Alert to mention: {state['weather_alert']}
    """
    
    structured_compiler = llm.with_structured_output(ItineraryOutput)
    response = structured_compiler.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=state['user_request'])])
    
    map_dicts = [{"name": loc.name, "lat": loc.lat, "lon": loc.lon} for loc in response.map_markers]
    
    return {
        "final_itinerary": response.markdown_text,
        "map_data": map_dicts
    }

# --- GRAPH ROUTING ---
def router(state: TravelState):
    if state.get('budget') is None:
        return "ask_human"
    return ["fetch_bookings", "fetch_tips", "check_weather"]

# --- BUILD GRAPH ---
workflow = StateGraph(TravelState)

workflow.add_node("compact_context", context_compactor)
workflow.add_node("analyze_intent", intent_analyzer)
workflow.add_node("fetch_bookings", fetch_bookings)
workflow.add_node("fetch_tips", fetch_tips)
workflow.add_node("check_weather", check_weather)
workflow.add_node("compile_itinerary", compile_itinerary)

workflow.set_entry_point("compact_context")
workflow.add_edge("compact_context", "analyze_intent")

workflow.add_conditional_edges(
    "analyze_intent",
    router,
    {
        "ask_human": END,
        "fetch_bookings": "fetch_bookings",
        "fetch_tips": "fetch_tips",
        "check_weather": "check_weather"
    }
)

workflow.add_edge(["fetch_bookings", "fetch_tips", "check_weather"], "compile_itinerary")
workflow.add_edge("compile_itinerary", END)

memory = MemorySaver()
travel_app = workflow.compile(checkpointer=memory)

def run_travel_agent(user_request: str, thread_id: str = "1", current_state: dict = None):
    config = {"configurable": {"thread_id": thread_id}}
    if current_state:
        travel_app.update_state(config, current_state)
    result = travel_app.invoke({"user_request": user_request}, config=config)
    return result