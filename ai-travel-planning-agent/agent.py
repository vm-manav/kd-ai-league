import os
import re
from datetime import datetime
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
    origin_city: Optional[str] = Field(default=None, description="Origin city for flights.")
    destination: str = Field(description="The city to travel to.")
    start_date: Optional[str] = Field(default=None, description="Trip start date (any human-readable format).")
    end_date: Optional[str] = Field(default=None, description="Trip end date (any human-readable format).")
    travelers: Optional[int] = Field(default=None, description="Number of travelers.")
    budget: Optional[int] = Field(description="Total budget in INR. Return None if not specified.")
    travel_style: Literal["Luxury", "Backpacker", "Family", "Adventure", "Standard"] = Field(...)
    preferences: List[str] = Field(default_factory=list, description="Optional preferences (food, pace, interests).")
    must_do: List[str] = Field(default_factory=list, description="Must-do activities or constraints.")

class BookingAgentOutput(BaseModel):
    flights_summary: str = Field(description="One-line flight summary with price in INR.")
    hotels_summary: str = Field(description="One-line hotel summary with price per night in INR.")
    total_estimated_inr: Optional[int] = Field(default=None, description="Rough total for flights + 3 nights if derivable.")

class PlanOutput(BaseModel):
    plan_markdown: str = Field(description="Structured trip plan in markdown (sections, bullets).")
    assumptions: List[str] = Field(description="Key assumptions used for planning.")
    open_questions: List[str] = Field(description="Open questions that require user confirmation.")
    approval_prompt: str = Field(description="Short approval prompt asking user to confirm or edit the plan.")

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
    hotel_names: List[str] = Field(description="List of 1-3 hotel names included or recommended.")

class ItineraryVariant(BaseModel):
    title: str = Field(description="Short name for the variant (e.g., Foodie, Adventure, Culture).")
    markdown_text: str = Field(description="Formatted markdown itinerary for this variant.")
    map_markers: List[MapLocation] = Field(description="List of 3-5 key locations for the map")
    hotel_names: List[str] = Field(description="List of 1-3 hotel names for this variant.")

class ItinerarySetOutput(BaseModel):
    variants: List[ItineraryVariant] = Field(description="2-3 itinerary variants within the same dates/budget/place.")

class BookingReadyOutput(BaseModel):
    booking_ready_markdown: str = Field(description="Booking-ready package in markdown.")
    checklist: List[str] = Field(description="Short checklist of items required to book.")
    risks: List[str] = Field(description="Potential risks or constraints to confirm before booking.")

class BookingReadyVariant(BaseModel):
    title: str = Field(description="Variant title.")
    booking_ready_markdown: str = Field(description="Booking-ready package for this variant.")
    checklist: List[str] = Field(description="Checklist items.")
    risks: List[str] = Field(description="Risks to confirm.")

class BookingReadySetOutput(BaseModel):
    variants: List[BookingReadyVariant] = Field(description="Booking-ready packages for each itinerary variant.")

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
    origin_city: Optional[str]
    destination: str
    start_date: Optional[str]
    end_date: Optional[str]
    travelers: Optional[int]
    budget: Optional[int]
    travel_style: str
    preferences: List[str]
    must_do: List[str]
    missing_fields: List[str]
    plan_approved: bool
    # Structured agent outputs (dict form for state)
    plan_output: Optional[dict]
    booking_output: Optional[dict]
    tips_output: Optional[dict]
    weather_output: Optional[dict]
    # Display strings (kept for UI and compiler)
    plan_markdown: str
    flights_data: str
    hotels_data: str
    insider_tips: str
    weather_alert: str
    final_itinerary: str
    map_data: List[dict]
    map_links: List[dict]
    hotel_links: List[dict]
    booking_ready: str
    booking_ready_variants: List[dict]
    itinerary_variants: List[dict]
    missing_info_question: str
    summary: str
    validation_issues: Optional[str]             
    logs: Annotated[List[dict], operator.add]
    evidence: Annotated[List[dict], operator.add]
    task_status: Annotated[dict, operator.or_]
    current_step: Annotated[List[str], operator.add]
    changed_fields: List[str]
    replan_required: bool
    prev_snapshot: dict
    current_snapshot: dict
    last_log_id: int

# Simple caches to reduce repeated tool calls
TIPS_CACHE = {}
WEATHER_CACHE = {}

def _parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    patterns = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None

def _missing_fields_message(missing: List[str]) -> str:
    if not missing:
        return ""
    friendly = {
        "origin_city": "origin city",
        "start_date": "start date (e.g., 12 Mar 2026)",
        "end_date": "end date (e.g., 15 Mar 2026)",
        "travelers": "number of travelers",
        "budget": "budget in INR",
        "destination": "destination",
    }
    items = [friendly.get(m, m) for m in missing]
    return "To proceed, please share: " + ", ".join(items) + "."

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
    prev_dest = state.get("destination", "")
    prev_style = state.get("travel_style", "")
    prev_origin = state.get("origin_city", "")
    prev_dates = (state.get("start_date", ""), state.get("end_date", ""))
    prev_travelers = state.get("travelers", None)
    context = (
        "Extract trip intent from the LATEST user message. "
        "If the latest message explicitly names a destination, use it. "
        "If the latest message does NOT mention any destination, you may reuse the previous destination "
        "provided below. Do not invent a new city.\n\n"
        f"Latest user message:\n{latest}\n\n"
        f"Previous destination (use only if none in latest): {prev_dest}\n"
        f"Previous travel style (use only if none in latest): {prev_style}\n"
        f"Previous origin (use only if none in latest): {prev_origin}\n"
        f"Previous dates (use only if none in latest): {prev_dates[0]} to {prev_dates[1]}\n"
        f"Previous travelers (use only if none in latest): {prev_travelers}"
    )
    if state.get("summary"):
        context += f"\n\n(Previous context for budget only if not in latest message: {state.get('summary')})"
    intent = structured_llm.invoke(context)

    # Fill missing fields from previous state if not in latest message
    effective_origin = intent.origin_city or state.get("origin_city")
    effective_dest = intent.destination or state.get("destination")
    effective_start = intent.start_date or state.get("start_date")
    effective_end = intent.end_date or state.get("end_date")
    effective_travelers = intent.travelers if intent.travelers is not None else state.get("travelers")
    effective_budget = intent.budget if intent.budget is not None else state.get("budget")
    effective_style = intent.travel_style or state.get("travel_style")
    effective_prefs = intent.preferences or state.get("preferences", [])
    effective_must_do = intent.must_do or state.get("must_do", [])

    def _norm(s):
        return (s or "").strip().lower()

    prev_snapshot = {
        "origin_city": state.get("origin_city"),
        "destination": state.get("destination"),
        "start_date": state.get("start_date"),
        "end_date": state.get("end_date"),
        "travelers": state.get("travelers"),
        "budget": state.get("budget"),
        "travel_style": state.get("travel_style"),
        "preferences": state.get("preferences", []),
        "must_do": state.get("must_do", []),
    }
    current_snapshot = {
        "origin_city": effective_origin,
        "destination": effective_dest,
        "start_date": effective_start,
        "end_date": effective_end,
        "travelers": effective_travelers,
        "budget": effective_budget,
        "travel_style": effective_style,
        "preferences": effective_prefs,
        "must_do": effective_must_do,
    }

    changed_fields = []
    if _norm(state.get("origin_city")) != _norm(effective_origin) and effective_origin:
        changed_fields.append("origin_city")
    if _norm(state.get("destination")) != _norm(effective_dest) and effective_dest:
        changed_fields.append("destination")
    if _norm(state.get("start_date")) != _norm(effective_start) and effective_start:
        changed_fields.append("start_date")
    if _norm(state.get("end_date")) != _norm(effective_end) and effective_end:
        changed_fields.append("end_date")
    if state.get("travelers") != effective_travelers and effective_travelers is not None:
        changed_fields.append("travelers")
    if state.get("budget") != effective_budget and effective_budget is not None:
        changed_fields.append("budget")
    if _norm(state.get("travel_style")) != _norm(effective_style) and effective_style:
        changed_fields.append("travel_style")
    if state.get("preferences", []) != effective_prefs and effective_prefs:
        changed_fields.append("preferences")
    if state.get("must_do", []) != effective_must_do and effective_must_do:
        changed_fields.append("must_do")

    replan_required = any(
        f in changed_fields
        for f in ["origin_city", "destination", "start_date", "end_date", "travelers", "budget", "travel_style"]
    )

    missing = []
    if not effective_dest:
        missing.append("destination")
    if effective_budget is None:
        missing.append("budget")
    if not effective_start:
        missing.append("start_date")
    if not effective_end:
        missing.append("end_date")
    if effective_travelers is None:
        missing.append("travelers")
    if not effective_origin:
        missing.append("origin_city")
    question = _missing_fields_message(missing)

    log = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "agent": "intent_analyzer",
        "payload": {
            "origin_city": effective_origin,
            "destination": effective_dest,
            "start_date": effective_start,
            "end_date": effective_end,
            "travelers": effective_travelers,
            "budget": effective_budget,
            "travel_style": effective_style,
            "preferences": effective_prefs,
            "must_do": effective_must_do,
            "missing_fields": missing,
            "changed_fields": changed_fields,
            "replan_required": replan_required,
            "prev_snapshot": prev_snapshot,
            "current_snapshot": current_snapshot,
        },
    }
    return {
        "origin_city": effective_origin,
        "destination": effective_dest,
        "start_date": effective_start,
        "end_date": effective_end,
        "travelers": effective_travelers,
        "budget": effective_budget,
        "travel_style": effective_style,
        "preferences": effective_prefs,
        "must_do": effective_must_do,
        "missing_fields": missing,
        "missing_info_question": question,
        "changed_fields": changed_fields,
        "replan_required": replan_required,
        "prev_snapshot": prev_snapshot,
        "current_snapshot": current_snapshot,
        "task_status": {"intent": "done"},
        "current_step": ["intent"],
        "logs": [log],
    }

def create_plan(state: TravelState):
    print("🧭 Creating Structured Trip Plan...")
    structured_llm = llm.with_structured_output(PlanOutput)
    prompt = (
        "Create a structured travel plan proposal. Include sections: Overview, Assumptions, "
        "Day-by-day outline (high level), Budget allocation (rough), Open questions, "
        "Approval prompt. Keep it concise and in markdown."
        f"\n\nUser request: {state.get('user_request','')}. "
        f"Destination: {state.get('destination','')}. Style: {state.get('travel_style','Standard')}. "
        f"Budget: {state.get('budget')}. "
        f"Dates: {state.get('start_date')} to {state.get('end_date')}. "
        f"Origin: {state.get('origin_city')}. Travelers: {state.get('travelers')}. "
        f"Preferences: {', '.join(state.get('preferences', []))}. "
        f"Must-do: {', '.join(state.get('must_do', []))}."
    )
    out = structured_llm.invoke(prompt)
    payload = out.model_dump()
    log = {"ts": datetime.utcnow().isoformat() + "Z", "agent": "create_plan", "payload": payload}
    return {
        "plan_output": payload,
        "plan_markdown": out.plan_markdown,
        "missing_info_question": out.approval_prompt,
        "plan_approved": False,
        "task_status": {"plan": "done"},
        "current_step": ["plan"],
        "logs": [log],
    }

def fetch_bookings(state: TravelState):
    print("✈️ [Worker 1] Fetching Flights & Hotels...")
    dest = state.get("destination", "city")
    origin = state.get("origin_city", "origin city")
    dates = f"{state.get('start_date')} to {state.get('end_date')}"
    travelers = state.get("travelers", 1)
    # Tool-stubbed structured outputs (deterministic)
    flights_data = f"{origin} → {dest} | {dates} | {travelers} pax | Estimated INR 12,000 per person"
    hotels_data = f"{dest} | 3-night stay | Estimated INR 4,000 per night (mid-range)"
    payload = BookingAgentOutput(
        flights_summary=flights_data,
        hotels_summary=hotels_data,
        total_estimated_inr=12000 * max(1, int(travelers)) + 4000 * 3,
    ).model_dump()
    log = {"ts": datetime.utcnow().isoformat() + "Z", "agent": "fetch_bookings", "payload": payload}
    return {
        "booking_output": payload,
        "flights_data": flights_data,
        "hotels_data": hotels_data,
        "task_status": {"bookings": "done"},
        "current_step": ["bookings"],
        "evidence": [{"agent": "fetch_bookings", "type": "stub", "details": payload}],
        "logs": [log],
    }

def fetch_tips(state: TravelState):
    print("🕵️ [Worker 2] Scouring Reddit for secrets...")
    cache_key = (state.get("destination", ""), state.get("travel_style", "Standard"))
    if cache_key in TIPS_CACHE:
        return TIPS_CACHE[cache_key]
    search = TavilySearchResults(max_results=2)
    query = f"site:reddit.com best {state['travel_style']} food in {state['destination']}"
    results = search.invoke({"query": query})
    raw = "\n".join([f"- {r['content']}" for r in results])
    structured_llm = llm.with_structured_output(LocalTipsOutput)
    tips_list = structured_llm.invoke(f"From these notes, extract 2-5 short insider tips as a list. Notes:\n{raw}")
    payload = tips_list.model_dump()
    insider_tips = "\n".join([f"- {t}" for t in tips_list.tips]) if tips_list.tips else raw
    log = {"ts": datetime.utcnow().isoformat() + "Z", "agent": "fetch_tips", "payload": payload}
    evidence = [{"agent": "fetch_tips", "type": "search", "details": results}]
    result = {
        "tips_output": payload,
        "insider_tips": insider_tips,
        "task_status": {"tips": "done"},
        "current_step": ["tips"],
        "evidence": evidence,
        "logs": [log],
    }
    TIPS_CACHE[cache_key] = result
    return result

def check_weather(state: TravelState):
    print("🌤️ [Worker 3] Checking Weather & Disasters...")
    cache_key = state.get("destination", "").lower()
    if cache_key in WEATHER_CACHE:
        return WEATHER_CACHE[cache_key]
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
    log = {"ts": datetime.utcnow().isoformat() + "Z", "agent": "check_weather", "payload": payload}
    result = {
        "weather_output": payload,
        "weather_alert": out.message,
        "task_status": {"weather": "done"},
        "current_step": ["weather"],
        "evidence": [{"agent": "check_weather", "type": "rule", "details": payload}],
        "logs": [log],
    }
    WEATHER_CACHE[cache_key] = result
    return result

def compile_itinerary(state: TravelState):
    print("📝 Compiling Final Itinerary & Map Coordinates...")
    sys_prompt = f"""You are an elite Travel AI. Create 2-3 distinct itinerary variants for the same trip.
    Each variant must have a different theme (e.g., culture, adventure, food, relaxation) but keep the same dates, budget, and destination.
    Use clean markdown styling: H2/H3 headings, bold section labels, concise bullet lists, and a compact day-by-day table.
    Include a "Summary" block, "Daily Plan" table, "Food & Local Spots", "Logistics", and "Budget Snapshot" sections.
    Budget: ₹{state['budget']}
    Dates: {state.get('start_date')} to {state.get('end_date')}
    Travelers: {state.get('travelers')}
    Origin: {state.get('origin_city')}
    Flights: {state['flights_data']}
    Hotels: {state['hotels_data']}
    Reddit Tips: {state['insider_tips']}
    Weather Alert to mention: {state['weather_alert']}
    """
    structured_compiler = llm.with_structured_output(ItinerarySetOutput)
    response = structured_compiler.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=state["user_request"])])

    def _valid_latlon(lat: float, lon: float) -> bool:
        try:
            return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0
        except Exception:
            return False

    def _maps_link(name: str) -> dict:
        q = name.replace(" ", "+")
        return {"name": name, "url": f"https://www.google.com/maps/search/?api=1&query={q}"}

    variants_payload = []
    for v in response.variants:
        map_dicts = [
            {"name": loc.name, "lat": loc.lat, "lon": loc.lon}
            for loc in v.map_markers
            if _valid_latlon(loc.lat, loc.lon)
        ]
        map_links = [_maps_link(loc.name) for loc in v.map_markers]
        hotel_links = [{"name": h, "url": f"https://www.google.com/maps/search/?api=1&query={h.replace(' ', '+')}"} for h in v.hotel_names]
        variants_payload.append(
            {
                "title": v.title,
                "markdown_text": v.markdown_text,
                "map_data": map_dicts,
                "map_links": map_links,
                "hotel_links": hotel_links,
            }
        )

    # Use the first variant as the default main itinerary for existing UI sections
    first = variants_payload[0] if variants_payload else {}
    log = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "agent": "compile_itinerary",
        "payload": {"variants": [v.get("title") for v in variants_payload]},
    }
    return {
        "final_itinerary": first.get("markdown_text", ""),
        "map_data": first.get("map_data", []),
        "map_links": first.get("map_links", []),
        "hotel_links": first.get("hotel_links", []),
        "itinerary_variants": variants_payload,
        "task_status": {"compile": "done"},
        "current_step": ["compile"],
        "logs": [log],
    }

def build_booking_ready_package(state: TravelState):
    print("📦 Building booking-ready package...")
    variants = state.get("itinerary_variants") or []
    if variants:
        structured_llm = llm.with_structured_output(BookingReadySetOutput)
        compact = []
        for v in variants:
            compact.append(
                {
                    "title": v.get("title", ""),
                    "hotels": [h.get("name") for h in (v.get("hotel_links") or [])],
                    "itinerary": (v.get("markdown_text") or "")[:800],
                }
            )
        prompt = (
            "Create a booking-ready package for each variant. Include traveler details needed, "
            "selected flight/hotel summaries, price assumptions, and next actions. Provide a short checklist "
            "and any booking risks to confirm.\n\n"
            f"Flights: {state.get('flights_data','')}.\n"
            f"Budget: {state.get('budget')}.\n"
            f"Dates: {state.get('start_date')} to {state.get('end_date')}.\n"
            f"Travelers: {state.get('travelers')}.\n"
            f"Origin: {state.get('origin_city')}.\n"
            f"Variants: {compact}"
        )
        out = structured_llm.invoke(prompt)
        items = []
        for v in out.variants:
            checklist_md = "\n".join([f"- {c}" for c in v.checklist]) if v.checklist else "- None"
            risks_md = "\n".join([f"- {r}" for r in v.risks]) if v.risks else "- None"
            booking_ready = (
                f"{v.booking_ready_markdown}\n\n"
                f"**Booking checklist**\n{checklist_md}\n\n"
                f"**Risks to confirm**\n{risks_md}"
            )
            items.append({"title": v.title, "booking_ready": booking_ready})
        first = items[0]["booking_ready"] if items else ""
        log = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "agent": "build_booking_ready_package",
            "payload": {"variants": [v.get("title") for v in items]},
        }
        return {
            "booking_ready_variants": items,
            "booking_ready": first,
            "task_status": {"booking_ready": "done"},
            "current_step": ["booking_ready"],
            "logs": [log],
        }

    structured_llm = llm.with_structured_output(BookingReadyOutput)
    prompt = (
        "Create a booking-ready package in markdown with: confirmed traveler details needed, "
        "selected flight/hotel summaries, price assumptions, and next actions. Provide a short checklist "
        "and any booking risks to confirm.\n\n"
        f"Flights: {state.get('flights_data','')}.\n"
        f"Hotels: {state.get('hotels_data','')}.\n"
        f"Budget: {state.get('budget')}.\n"
        f"Dates: {state.get('start_date')} to {state.get('end_date')}.\n"
        f"Travelers: {state.get('travelers')}.\n"
        f"Origin: {state.get('origin_city')}.\n"
        f"Itinerary: {state.get('final_itinerary','')[:1200]}"
    )
    out = structured_llm.invoke(prompt)
    checklist_md = "\n".join([f"- {c}" for c in out.checklist]) if out.checklist else "- None"
    risks_md = "\n".join([f"- {r}" for r in out.risks]) if out.risks else "- None"
    booking_ready = (
        f"{out.booking_ready_markdown}\n\n"
        f"**Booking checklist**\n{checklist_md}\n\n"
        f"**Risks to confirm**\n{risks_md}"
    )
    log = {"ts": datetime.utcnow().isoformat() + "Z", "agent": "build_booking_ready_package", "payload": out.model_dump()}
    return {
        "booking_ready": booking_ready,
        "task_status": {"booking_ready": "done"},
        "current_step": ["booking_ready"],
        "logs": [log],
    }


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
    # Missing/invalid dates
    start_dt = _parse_date(state.get("start_date") or "")
    end_dt = _parse_date(state.get("end_date") or "")
    missing_dates = start_dt is None or end_dt is None
    if missing_dates:
        issues.append("Trip dates are missing or not in a clear format.")
    elif end_dt < start_dt:
        issues.append("Trip end date is before the start date.")
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
    log = {"ts": datetime.utcnow().isoformat() + "Z", "agent": "itinerary_validator", "payload": result.model_dump()}
    return {
        "validation_issues": suggestion if not valid else None,
        "missing_info_question": suggestion or state.get("missing_info_question", ""),
        "task_status": {"validate": "done" if valid else "needs_attention"},
        "current_step": ["validate"],
        "logs": [log],
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
    if state.get("replan_required"):
        return "create_plan"
    if state.get("missing_fields"):
        return "ask_human"
    if state.get("plan_approved"):
        return ["fetch_bookings", "fetch_tips", "check_weather"]
    return "create_plan"

# --- BUILD GRAPH ---
workflow = StateGraph(TravelState)

workflow.add_node("compact_context", context_compactor)
workflow.add_node("analyze_intent", intent_analyzer)
workflow.add_node("fetch_bookings", fetch_bookings)
workflow.add_node("fetch_tips", fetch_tips)
workflow.add_node("check_weather", check_weather)
workflow.add_node("create_plan", create_plan)
workflow.add_node("compile_itinerary", compile_itinerary)
workflow.add_node("booking_ready", build_booking_ready_package)
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
        "create_plan": "create_plan",
    },
)

# Plan approval checkpoint
workflow.add_edge("create_plan", END)

# Parallel workers -> weather guardrail (recovery edge)
workflow.add_edge(["fetch_bookings", "fetch_tips", "check_weather"], "weather_guardrail")
workflow.add_conditional_edges(
    "weather_guardrail",
    weather_guardrail_router,
    {"ask_date_change": END, "compile": "compile_itinerary"},
)

# Compiler -> Validator -> (valid -> compact -> END | invalid -> END)
workflow.add_edge("compile_itinerary", "booking_ready")
workflow.add_edge("booking_ready", "itinerary_validator")
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

def reset_thread(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        travel_app.delete_state(config)
    except Exception:
        # If the backend does not support deletion, ignore.
        pass
