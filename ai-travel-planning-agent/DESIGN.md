# Autonomous Travel Swarm — Design Doc

## Overview
This project is an agentic travel planning system built with Streamlit for UI and LangGraph for orchestration. It supports multi‑turn planning, approval checkpoints, parallel execution, validation, and booking‑ready outputs. The system emphasizes transparency via logs, evidence, and replan diffs.

## Goals
- Provide structured trip planning with clear approval gates.
- Run parallel agents for bookings, tips, and weather.
- Validate outputs before delivery.
- Support multiple itinerary variants with comparison.
- Maintain observable state with logs and evidence.
- Support replanning on key input changes.

## Non‑Goals
- Actual bookings with live provider integrations (currently stubbed).
- Real-time availability and pricing verification.

## Architecture
### Layers
- **UI Layer**: Streamlit chat UI, variant compare, map, logs/evidence, workflow status.
- **Orchestration Layer**: LangGraph StateGraph with checkpoints and conditional routing.
- **Agent Layer**: Intent, planning, bookings, tips, weather, compiler, booking‑ready, validator.
- **Data Layer**: Session state, per‑chat trip data, logs, evidence, and snapshots.

### Primary Modules
- `agent.py`
  - Defines Pydantic schemas, TravelState, LangGraph nodes, routing logic.
- `app.py`
  - Streamlit UI, chat sessions, rendering of variants, maps, logs/evidence, workflow status.

## Workflow
1. **Intent Analysis**: Parse the latest user message, reuse prior values when not specified.
2. **Missing Info Gate**: Ask for missing required fields (destination, dates, budget, travelers, origin).
3. **Plan Proposal**: Generate a structured plan with a human approval prompt.
4. **Approval Checkpoint**: Proceed only after user approval.
5. **Parallel Execution**: Fetch bookings (stub), tips (Tavily), weather (rules).
6. **Compilation**: Generate 2–3 themed itinerary variants with maps and hotel links.
7. **Booking‑Ready Output**: Generate per‑variant booking readiness package.
8. **Validation**: Check budget, dates, weather conflicts.
9. **Completion**: Render final output and optional booking approval.

## Agent Flow (Detailed)
The graph executes in a deterministic sequence with explicit gates and parallel branches:

1. **context_compactor → intent_analyzer**
   - The compactor summarizes long histories to prevent context drift.
   - The intent analyzer extracts structured trip inputs and computes `changed_fields` + `replan_required`.

2. **Gate A: Missing Info**
   - If required fields are missing, the flow ends at `ask_human` and waits for user input.

3. **Gate B: Replan Required**
   - If critical fields changed, the flow routes to `create_plan` even if a previous plan exists.
   - The plan step always ends at a human approval checkpoint.

4. **Approval Checkpoint**
   - The user must approve the plan to proceed.

5. **Parallel Agents (Fan‑Out)**
   - `fetch_bookings` (deterministic stub; later replace with real provider)
   - `fetch_tips` (Tavily search)
   - `check_weather` (rule-based; later replace with live API)

6. **Weather Guardrail**
   - If severe, pause and ask the user to confirm new dates.
   - If ok, continue.

7. **Compiler**
   - `compile_itinerary` generates 2–3 themed variants, map points, and links.

8. **Booking‑Ready Packager**
   - `build_booking_ready_package` creates per‑variant booking checklists and risks.

9. **Validation & End**
   - `itinerary_validator` checks budget, dates, and weather conflicts.
   - On success, flow compacts state and ends; otherwise, returns issues for user recovery.

## State Model (TravelState)
Key fields:
- `user_request`, `origin_city`, `destination`, `start_date`, `end_date`, `travelers`, `budget`, `travel_style`
- `plan_markdown`, `plan_approved`
- `itinerary_variants`, `map_data`, `map_links`, `hotel_links`
- `booking_ready`, `booking_ready_variants`
- `task_status`, `current_step`
- `logs`, `evidence`, `prev_snapshot`, `current_snapshot`, `changed_fields`, `replan_required`

## Replanning
- `intent_analyzer` computes a diff between previous and current snapshots.
- If critical fields change (destination, dates, budget, travelers, origin, style), `replan_required = True`.
- Replan forces a new plan proposal and resets approval.

## Observability
- **Logs**: Each agent node emits a timestamped structured log.
- **Evidence**: Sources or rule outputs are captured as evidence entries.
- **UI Panels**: Logs, Evidence, Workflow status, Dependencies, and Replan diff are visible in sidebar.

## UI/UX
- Sidebar: Chats, Trip Summary, Workflow, Dependencies, Logs, Evidence, Replan Diff.
- Main: Chat, Plan review, Itinerary variants, Map, Booking‑ready output.
- Variant compare: side‑by‑side for up to 3, horizontal scroll for more.

## Validation Rules
- Date parsing and ordering checks.
- Budget vs. estimated total.
- Weather conflicts (severe alerts).

## Current Limitations
- Booking data is stubbed (not live inventory).
- Map coordinates are LLM‑generated and validated only by bounds.
- Weather is rule‑based, not live.

## Future Enhancements
- Integrate real booking providers (flights/hotels/activities).
- Live weather and safety alerts.
- Stronger map geocoding with real coordinates.
- Cost breakdown per day.
- Observability with metrics (latency, cost, tool errors).

## Dependencies
See `requirements.txt`.

## How to Run
```
source .venv/bin/activate
streamlit run app.py
```
