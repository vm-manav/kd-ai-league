import os
import re
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

# --- PYDANTIC MODELS (Structured outputs per agent) ---
class TripIntent(BaseModel):
    destination: str = Field(description="The city to travel to.")
    budget: Optional[int] = Field(description="Total budget in INR. Return None if not specified.")
    travel_style: Literal["Luxury", "Backpacker", "Family", "Adventure", "Standard"] = Field(...)

class BookingAgentOutput(BaseModel):
    flights_summary: str = Field(description="One-line flight summary with price in INR.")
    hotels_summary: str = Field(description="One-line hotel summary with price per night in INR.")
    total_estimated_inr: Optional[int] = Field(default=None, description="Rough total for flights + 3 nights if derivable.")

class LocalTipsOutput(BaseModel):
    tips: List[str] = Field(description="List of 2-5 short insider tips.")
    source: str = Field(default="reddit", description="Source e.g. reddit, local blogs.")

class WeatherAlertsOutput(BaseModel):
    severity: Literal["ok", "moderate", "severe"] = Field(description="ok = fine, moderate = pack accordingly, severe = consider date change.")
    message: str = Field(description="Short weather or disaster alert to show user.")
    recommend_date_change: bool = Field(default=False, description="True if severe and user should consider different dates.")

class MapLocation(BaseModel):
    name: str = Field(description="Name of the hotel or activity")
    lat: float = Field(description="Latitude")
    lon: float = Field(description="Longitude")

class ItineraryOutput(BaseModel):
    markdown_text: str = Field(description="The full formatted markdown itinerary")
    map_markers: List[MapLocation] = Field(description="List of 3-5 key locations for the map")

class ValidationResult(BaseModel):
    valid: bool = Field(description="True if itinerary is acceptable to deliver.")
    budget_exceeded: bool = Field(default=False, description="True if estimated cost exceeds user budget.")
    weather_conflict: bool = Field(default=False, description="True if itinerary conflicts with weather alert.")
    missing_dates: bool = Field(default=False, description="True if trip dates are not clear in itinerary.")
    incomplete_bookings: bool = Field(default=False, description="True if flights/hotels seem placeholder or missing.")
    issues: List[str] = Field(default_factory=list, description="Human-readable list of issues.")
    suggestion: Optional[str] = Field(default=None, description="Question or suggestion to ask user for recovery.")

# --- STATE (typed; structured outputs stored as dicts) ---
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_request: str
    destination: str
    budget: Optional[int]
    travel_style: str
    # Structured agent outputs (dict form for state)
    booking_output: Optional[dict]
    tips_output: Optional[dict]
    weather_output: Optional[dict]
    # Display strings (kept for UI and compiler)
    flights_data: str
    hotels_data: str
    insider_tips: str
    weather_alert: str
    final_itinerary: str
    map_data: List[dict]
    missing_info_question: str
    summary: str
    validation_issues: Optional[str]             

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
    latest = (state.get("user_request") or "").strip()
    # Use ONLY the latest user message for destination & travel_style so previous state (e.g. Goa) doesn't override (e.g. Jaipur)
    context = (
        "Extract trip intent from the LATEST user message only. "
        "Ignore any previous destination or city from earlier conversation. "
        "If the user mentions a city/place name, that is the destination for this request.\n\n"
        f"Latest user message:\n{latest}"
    )
    if state.get("summary"):
        context += f"\n\n(Previous context, for budget only if not in latest message: {state.get('summary')})"
    intent = structured_llm.invoke(context)

    question = ""
    if intent.budget is None:
        question = f"I'd love to plan your trip to {intent.destination}! What is your approximate budget in INR?"

    return {
        "destination": intent.destination,
        "budget": intent.budget,
        "travel_style": intent.travel_style,
        "missing_info_question": question,
    }

def fetch_bookings(state: TravelState):
    print("✈️ [Worker 1] Fetching Flights & Hotels...")
    structured_llm = llm.with_structured_output(BookingAgentOutput)
    dest = state.get("destination", "city")
    prompt = f"User wants to travel to {dest}, style: {state.get('travel_style', 'Standard')}. Return one-line flights summary and one-line hotels summary with INR prices. Optional total_estimated_inr."
    out = structured_llm.invoke(prompt)
    payload = out.model_dump()
    flights_data = out.flights_summary
    hotels_data = out.hotels_summary
    return {
        "booking_output": payload,
        "flights_data": flights_data,
        "hotels_data": hotels_data,
    }

def fetch_tips(state: TravelState):
    print("🕵️ [Worker 2] Scouring Reddit for secrets...")
    search = TavilySearchResults(max_results=2)
    query = f"site:reddit.com best {state['travel_style']} food in {state['destination']}"
    results = search.invoke({"query": query})
    raw = "\n".join([f"- {r['content']}" for r in results])
    structured_llm = llm.with_structured_output(LocalTipsOutput)
    tips_list = structured_llm.invoke(f"From these notes, extract 2-5 short insider tips as a list. Notes:\n{raw}")
    payload = tips_list.model_dump()
    insider_tips = "\n".join([f"- {t}" for t in tips_list.tips]) if tips_list.tips else raw
    return {"tips_output": payload, "insider_tips": insider_tips}

def check_weather(state: TravelState):
    print("🌤️ [Worker 3] Checking Weather & Disasters...")
    dest = state.get("destination", "").lower()
    if "kerala" in dest or "mumbai" in dest:
        out = WeatherAlertsOutput(
            severity="severe",
            message="⚠️ WARNING: Heavy monsoon rains expected. Consider different dates or indoor backup activities.",
            recommend_date_change=True,
        )
    elif "rishikesh" in dest:
        out = WeatherAlertsOutput(severity="ok", message="✅ Weather is clear. Perfect for river rafting.", recommend_date_change=False)
    else:
        out = WeatherAlertsOutput(severity="ok", message="✅ Weather is pleasant. Standard packing recommended.", recommend_date_change=False)
    payload = out.model_dump()
    return {"weather_output": payload, "weather_alert": out.message}

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
    response = structured_compiler.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=state["user_request"])])
    map_dicts = [{"name": loc.name, "lat": loc.lat, "lon": loc.lon} for loc in response.map_markers]
    return {"final_itinerary": response.markdown_text, "map_data": map_dicts}


def itinerary_validator(state: TravelState):
    """Validation layer before END: budget, weather conflict, missing dates, incomplete bookings."""
    print("🔍 Validating itinerary (budget, weather, dates, bookings)...")
    budget = state.get("budget")
    itinerary = state.get("final_itinerary") or ""
    weather_alert = state.get("weather_alert") or ""
    booking = state.get("booking_output") or {}
    issues = []
    budget_exceeded = False
    incomplete_bookings = False
    # Budget: try to parse INR from itinerary or use booking total
    total_estimated = booking.get("total_estimated_inr")
    if budget is not None and total_estimated is not None and total_estimated > budget:
        budget_exceeded = True
        issues.append(f"Estimated cost ₹{total_estimated} exceeds budget ₹{budget}.")
    # Missing dates: no clear date pattern
    date_pattern = re.search(r"\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}", itinerary, re.I)
    missing_dates = not bool(date_pattern)
    if missing_dates:
        issues.append("Trip dates are not clearly mentioned in the itinerary.")
    # Weather conflict: severe alert but itinerary doesn't acknowledge it
    weather_conflict = "severe" in (state.get("weather_output") or {}).get("severity", "") and "monsoon" in weather_alert.lower() and "indoor" not in itinerary.lower() and "rain" not in itinerary.lower()
    if weather_conflict:
        issues.append("Weather alert suggests date change but itinerary doesn't reflect it.")
    # Incomplete: placeholder-looking content
    incomplete_bookings = "Taj" in (state.get("hotels_data") or "") and "Vistara" in (state.get("flights_data") or "") and len(itinerary) < 200
    if incomplete_bookings:
        issues.append("Bookings look like placeholders; real availability not confirmed.")
    valid = not (budget_exceeded or weather_conflict or incomplete_bookings)
    suggestion = None
    if not valid and issues:
        suggestion = " ".join(issues) + " Would you like to adjust budget, dates, or shall I re-plan with these in mind?"
    result = ValidationResult(
        valid=valid,
        budget_exceeded=budget_exceeded,
        weather_conflict=weather_conflict,
        missing_dates=missing_dates,
        incomplete_bookings=incomplete_bookings,
        issues=issues,
        suggestion=suggestion,
    )
    return {
        "validation_issues": suggestion if not valid else None,
        "missing_info_question": suggestion or state.get("missing_info_question", ""),
    }


def compact_after_compile(state: TravelState):
    """Compaction loop after compiler: summarize, clear heavy intermediate data."""
    print("🗜️ Compacting after itinerary: storing summary, clearing heavy data...")
    messages = state.get("messages", [])
    summary = state.get("summary", "")
    new_summary = (
        f"Trip: {state.get('destination', '')} | {state.get('travel_style', '')} | Budget ₹{state.get('budget')}. "
        f"Itinerary generated. Weather: {state.get('weather_alert', '')[:80]}."
    )
    # Clear heavy intermediates; keep final_itinerary, map_data, weather_alert, summary
    return {
        "summary": new_summary,
        "flights_data": "",
        "hotels_data": "",
        "insider_tips": "",
        "booking_output": None,
        "tips_output": None,
    }


def weather_guardrail(state: TravelState):
    """Passthrough: sets missing_info_question if we need to ask for date change (recovery edge)."""
    wo = state.get("weather_output") or {}
    if wo.get("severity") == "severe" and wo.get("recommend_date_change"):
        return {"missing_info_question": f"{state.get('weather_alert', '')} Consider different travel dates and I'll re-plan."}
    return {}

def weather_guardrail_router(state: TravelState):
    """Error recovery: if weather is severe and recommends date change, ask user before compiling."""
    wo = state.get("weather_output") or {}
    if wo.get("severity") == "severe" and wo.get("recommend_date_change"):
        return "ask_date_change"
    return "compile"


# --- GRAPH ROUTING ---
def router(state: TravelState):
    if state.get("budget") is None:
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
workflow.add_node("weather_guardrail", weather_guardrail)
workflow.add_node("itinerary_validator", itinerary_validator)
workflow.add_node("compact_after_compile", compact_after_compile)

workflow.set_entry_point("compact_context")
workflow.add_edge("compact_context", "analyze_intent")

workflow.add_conditional_edges(
    "analyze_intent",
    router,
    {
        "ask_human": END,
        "fetch_bookings": "fetch_bookings",
        "fetch_tips": "fetch_tips",
        "check_weather": "check_weather",
    },
)

# Parallel workers -> weather guardrail (recovery edge)
workflow.add_edge(["fetch_bookings", "fetch_tips", "check_weather"], "weather_guardrail")
workflow.add_conditional_edges(
    "weather_guardrail",
    weather_guardrail_router,
    {"ask_date_change": END, "compile": "compile_itinerary"},
)

# Compiler -> Validator -> (valid -> compact -> END | invalid -> END)
workflow.add_edge("compile_itinerary", "itinerary_validator")
workflow.add_conditional_edges(
    "itinerary_validator",
    lambda s: "compact" if not s.get("validation_issues") else "reject",
    {"compact": "compact_after_compile", "reject": END},
)
workflow.add_edge("compact_after_compile", END)

memory = MemorySaver()
travel_app = workflow.compile(checkpointer=memory)

def run_travel_agent(user_request: str, thread_id: str = "1", current_state: dict = None):
    config = {"configurable": {"thread_id": thread_id}}
    if current_state:
        travel_app.update_state(config, current_state)
    result = travel_app.invoke({"user_request": user_request}, config=config)
    return result