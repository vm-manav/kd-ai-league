import streamlit as st
import folium
import uuid
import html
import re
import textwrap
import io
from xml.sax.saxutils import escape
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, ListFlowable, ListItem
from streamlit_folium import st_folium
from agent import run_travel_agent, reset_thread

st.set_page_config(page_title="Swarm Travel Planner", layout="wide", page_icon="🌍")

st.title("🌍 Autonomous Travel Swarm")
st.markdown("Powered by LangGraph Parallel Agents (Planning, Booking, Weather, Local Expert)")

# Initialize session state
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    new_id = str(uuid.uuid4())
    st.session_state.active_chat_id = new_id
    st.session_state.chats[new_id] = {
        "title": "New chat",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "thread_id": str(uuid.uuid4()),
        "chat_history": [],
        "log_history": [],
        "trip_data": None,
        "status": "planning",
        "plan_approved": False,
    }

with st.sidebar:
    st.subheader("Chats")
    chat_ids = list(st.session_state.chats.keys())
    labels = [
        f"{st.session_state.chats[c]['title']} · {st.session_state.chats[c]['created_at']}"
        for c in chat_ids
    ]
    if chat_ids:
        selected = st.radio(
            "Saved conversations",
            options=chat_ids,
            format_func=lambda cid: labels[chat_ids.index(cid)],
            index=chat_ids.index(st.session_state.active_chat_id),
        )
        if selected != st.session_state.active_chat_id:
            st.session_state.active_chat_id = selected
            st.rerun()

    if st.button("New chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.active_chat_id = new_id
        st.session_state.chats[new_id] = {
            "title": "New chat",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "thread_id": str(uuid.uuid4()),
            "chat_history": [],
            "log_history": [],
            "trip_data": None,
            "status": "planning",
            "plan_approved": False,
        }
        st.rerun()

    if st.button("Clear current chat", use_container_width=True):
        active = st.session_state.active_chat_id
        thread_id = st.session_state.chats[active]["thread_id"]
        reset_thread(thread_id)
        st.session_state.chats[active]["chat_history"] = []
        st.session_state.chats[active]["log_history"] = []
        st.session_state.chats[active]["trip_data"] = None
        st.session_state.chats[active]["status"] = "planning"
        st.session_state.chats[active]["plan_approved"] = False
        st.rerun()

active = st.session_state.chats[st.session_state.active_chat_id]

# Trip summary
with st.sidebar:
    st.subheader("Trip Summary")
    data = active.get("trip_data") or {}
    summary_lines = []
    if data.get("destination"):
        summary_lines.append(f"Destination: {data.get('destination')}")
    if data.get("origin_city"):
        summary_lines.append(f"Origin: {data.get('origin_city')}")
    if data.get("start_date") or data.get("end_date"):
        summary_lines.append(f"Dates: {data.get('start_date')} to {data.get('end_date')}")
    if data.get("travelers") is not None:
        summary_lines.append(f"Travelers: {data.get('travelers')}")
    if data.get("budget") is not None:
        summary_lines.append(f"Budget: {data.get('budget')} INR")
    if data.get("travel_style"):
        summary_lines.append(f"Style: {data.get('travel_style')}")
    if summary_lines:
        st.write("\n".join(summary_lines))
    else:
        st.caption("No trip details yet.")

# Workflow status
with st.sidebar:
    st.subheader("Workflow")
    data = active.get("trip_data") or {}
    task_status = data.get("task_status", {})
    steps_list = data.get("current_step", [])
    current_step = steps_list[-1] if steps_list else "idle"
    steps = [
        "intent",
        "plan",
        "bookings",
        "tips",
        "weather",
        "compile",
        "booking_ready",
        "validate",
    ]
    for s in steps:
        st.write(f"{s}: {task_status.get(s, 'pending')}")
    st.caption(f"Current step: {current_step}")

# Task dependencies
with st.sidebar:
    st.subheader("Dependencies")
    st.write("plan -> bookings, tips, weather")
    st.write("bookings + tips + weather -> compile")
    st.write("compile -> booking_ready -> validate")

# Logs
with st.sidebar:
    st.subheader("LOGS")
    logs = active.get("log_history", [])
    if logs:
        log_text = "\n\n".join(
            [f"{l.get('ts','')} [{l.get('agent','agent')}] {l.get('payload')}" for l in logs]
        )
        st.markdown(
            f"""
<div style="max-height: 260px; overflow-y: auto; border: 1px solid #2a2f3a; border-radius: 10px; padding: 10px; background: #151a22;">
  <pre style="white-space: pre-wrap; font-size: 12px; margin: 0;">{log_text}</pre>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No logs yet.")

# Evidence
with st.sidebar:
    st.subheader("Evidence")
    evidence = (active.get("trip_data") or {}).get("evidence", [])
    if evidence:
        ev_text = "\n\n".join([f"[{e.get('agent','agent')}] {e.get('type','')} {e.get('details')}" for e in evidence])
        st.markdown(
            f"""
<div style="max-height: 200px; overflow-y: auto; border: 1px solid #2a2f3a; border-radius: 10px; padding: 10px; background: #151a22;">
  <pre style="white-space: pre-wrap; font-size: 12px; margin: 0;">{ev_text}</pre>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("No evidence yet.")

# Replan diff
with st.sidebar:
    st.subheader("Replan Diff")
    changed = (active.get("trip_data") or {}).get("changed_fields", [])
    if changed:
        st.write("Changed fields:")
        for f in changed:
            st.write(f"- {f}")
    else:
        st.caption("No changes detected.")

# Replan Diff (old vs new)
with st.sidebar:
    st.subheader("Replan Diff (Old → New)")
    data = active.get("trip_data") or {}
    prev_snap = data.get("prev_snapshot", {})
    curr_snap = data.get("current_snapshot", {})
    if prev_snap and curr_snap:
        rows = []
        for k in curr_snap.keys():
            if prev_snap.get(k) != curr_snap.get(k):
                rows.append({"field": k, "from": prev_snap.get(k), "to": curr_snap.get(k)})
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No field-level changes.")
    else:
        st.caption("No snapshots yet.")

# Chat Interface
for msg in active["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("E.g., Plan a luxury trip to Jaipur next week")

if user_input:
    active["chat_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Agents are executing in parallel..."):
        result = run_travel_agent(user_input, thread_id=active["thread_id"])
        active["trip_data"] = result
        if result.get("logs"):
            active["log_history"].extend(result.get("logs"))
        if result.get("replan_required"):
            active["plan_approved"] = False

        if result.get("plan_markdown") and not active.get("plan_approved"):
            active["status"] = "plan_ready"
            if active["title"] == "New chat":
                active["title"] = f"Plan: {result.get('destination','Trip')}"

        if result.get("missing_info_question") and not result.get("plan_markdown"):
            msg = result["missing_info_question"]
            active["chat_history"].append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
            result["missing_info_question"] = ""
            st.stop()

    if result.get("final_itinerary"):
        if result.get("validation_issues"):
            active["chat_history"].append({"role": "assistant", "content": "Trip draft ready but some issues were found. See validation below."})
            with st.chat_message("assistant"):
                st.markdown("Trip draft ready but some issues were found. See validation below.")
        else:
            active["chat_history"].append({"role": "assistant", "content": "Trip finalized! See details below."})
            with st.chat_message("assistant"):
                st.markdown("Trip finalized! Review your package below.")

# Render Dashboard
data = active.get("trip_data") or {}
if data:
    if data.get("plan_markdown") and not active.get("plan_approved"):
        st.divider()
        st.subheader("Plan Proposal (Review & Approve)")
        st.markdown(data["plan_markdown"])
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Approve plan", type="primary", use_container_width=True):
                active["plan_approved"] = True
                active["status"] = "executing"
                dest = data.get("destination", "the destination")
                style = data.get("travel_style", "Standard")
                budget = data.get("budget")
                budget_text = f" with budget {budget} INR" if budget is not None else ""
                followup = (
                    f"Approved. Proceed with bookings, weather check, and full itinerary for {dest} "
                    f"in {style} style{budget_text}."
                )
                active["chat_history"].append({"role": "user", "content": followup})
                with st.spinner("Proceeding with plan..."):
                    result = run_travel_agent(
                        followup,
                        thread_id=active["thread_id"],
                        current_state={"plan_approved": True},
                    )
                    active["trip_data"] = result
                    if result.get("logs"):
                        active["log_history"].extend(result.get("logs"))
                st.rerun()
        with c2:
            if st.button("Request edits", use_container_width=True):
                st.info("Tell me what to change in the chat box (dates, budget, pace, activities).")

    if data.get("final_itinerary") or data.get("itinerary_variants"):
        st.divider()

        if data.get("validation_issues"):
            st.warning("**Validation issues:** " + data["validation_issues"])
        if "WARNING" in data.get("weather_alert", "") or (data.get("weather_output") or {}).get("severity") == "severe":
            st.error(data.get("weather_alert", "Weather alert"))
        else:
            st.success(data.get("weather_alert", "Weather checked successfully."))

        col1, col2 = st.columns([2, 1.5])
    
        with col1:
            st.subheader("🗓️ Execution Package")
            chosen_title = None
            if data.get("itinerary_variants"):
                labels = [v.get("title", f"Variant {i+1}") for i, v in enumerate(data["itinerary_variants"])]
                compare_mode = st.toggle("Compare variants", value=False)
                if compare_mode:
                    sel = st.multiselect(
                        "Select variants to compare",
                        options=list(range(len(labels))),
                        format_func=lambda i: labels[i],
                        default=list(range(min(2, len(labels)))),
                    )
                    if sel:
                        focus_key = f"compare_focus_{st.session_state.active_chat_id}"
                        if focus_key not in st.session_state:
                            st.session_state[focus_key] = sel[0]
                        if len(sel) <= 3:
                            cols = st.columns(len(sel))
                            for col, i in zip(cols, sel):
                                v = data["itinerary_variants"][i]
                                with col:
                                    st.markdown(f"### {labels[i]}")
                                    if st.button(
                                        "Use for map/links",
                                        key=f"focus_{st.session_state.active_chat_id}_{i}",
                                        use_container_width=True,
                                    ):
                                        st.session_state[focus_key] = i
                                    st.markdown(v.get("markdown_text", ""))
                        else:
                            st.caption("Horizontal scroll enabled for more than 3 variants.")
                            focus = st.selectbox(
                                "Map/links based on",
                                options=sel,
                                format_func=lambda i: labels[i],
                                index=sel.index(st.session_state[focus_key]) if st.session_state[focus_key] in sel else 0,
                            )
                            st.session_state[focus_key] = focus
                            cards = []
                            for i in sel:
                                v = data["itinerary_variants"][i]
                                title = html.escape(labels[i])
                                body = html.escape(v.get("markdown_text", "")).replace("\n", "<br>")
                                cards.append(
                                    f"""
<div class="variant-card">
  <div class="variant-title">{title}</div>
  <div class="variant-body">{body}</div>
</div>
"""
                                )
                            st.markdown(
                                """
<style>
.variant-scroll{display:flex;gap:16px;overflow-x:auto;padding:6px 2px;}
.variant-card{min-width:320px;max-width:360px;flex:0 0 auto;border:1px solid #2a2f3a;border-radius:10px;padding:12px;background:#151a22;}
.variant-title{font-size:1.05rem;font-weight:600;margin-bottom:8px;}
.variant-body{font-size:0.9rem;line-height:1.4;color:#d0d6e0;}
</style>
<div class="variant-scroll">
"""
                                + "\n".join(cards)
                                + "\n</div>",
                                unsafe_allow_html=True,
                            )
                    if sel:
                        chosen = data["itinerary_variants"][st.session_state[focus_key]]
                    else:
                        chosen = data["itinerary_variants"][0]
                    chosen_title = chosen.get("title")
                else:
                    idx = st.selectbox("Choose itinerary variant", list(range(len(labels))), format_func=lambda i: labels[i])
                    chosen = data["itinerary_variants"][idx]
                    chosen_title = chosen.get("title")
                    st.markdown(chosen.get("markdown_text", ""))
            else:
                chosen = {
                    "markdown_text": data.get("final_itinerary", ""),
                    "map_data": data.get("map_data", []),
                    "map_links": data.get("map_links", []),
                    "hotel_links": data.get("hotel_links", []),
                }
                st.markdown(chosen["markdown_text"])
            
            def _sanitize_text(text: str) -> str:
                text = text.replace("₹", "INR ")
                text = text.encode("latin-1", "replace").decode("latin-1")
                return text

            def _break_long_words(text: str, max_len: int = 30) -> str:
                def _chunk(word: str) -> str:
                    return " ".join([word[i : i + max_len] for i in range(0, len(word), max_len)])
                return " ".join(_chunk(w) for w in text.split())

            def _to_pdf_bytes(md: str) -> bytes:
                def is_heading(ln: str) -> bool:
                    return ln.startswith("# ")

                def is_heading2(ln: str) -> bool:
                    return ln.startswith("## ")

                def is_heading3(ln: str) -> bool:
                    return ln.startswith("### ")

                def is_bullet(ln: str) -> bool:
                    return ln.lstrip().startswith(("- ", "• "))

                def is_table_row(ln: str) -> bool:
                    return "|" in ln

                lines = md.splitlines()
                i = 0
                blocks: list[tuple[str, object]] = []
                while i < len(lines):
                    line = lines[i].rstrip()
                    if not line.strip():
                        i += 1
                        continue
                    if is_heading(line):
                        blocks.append(("h1", line[2:].strip()))
                        i += 1
                        continue
                    if is_heading2(line):
                        blocks.append(("h2", line[3:].strip()))
                        i += 1
                        continue
                    if is_heading3(line):
                        blocks.append(("h3", line[4:].strip()))
                        i += 1
                        continue
                    if is_bullet(line):
                        items = []
                        while i < len(lines) and is_bullet(lines[i]):
                            items.append(lines[i].lstrip()[2:].strip())
                            i += 1
                        blocks.append(("ul", items))
                        continue
                    if is_table_row(line):
                        table_rows = []
                        while i < len(lines) and is_table_row(lines[i]):
                            row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                            if not all(set(c) <= {"-", ":"} for c in row if c):
                                table_rows.append(row)
                            i += 1
                        blocks.append(("table", table_rows))
                        continue
                    para = [line.strip()]
                    i += 1
                    while i < len(lines):
                        nxt = lines[i].rstrip()
                        if not nxt.strip() or is_heading(nxt) or is_heading2(nxt) or is_heading3(nxt) or is_bullet(nxt) or is_table_row(nxt):
                            break
                        para.append(nxt.strip())
                        i += 1
                    blocks.append(("p", " ".join(para)))

                buffer = io.BytesIO()
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=A4,
                    leftMargin=36,
                    rightMargin=36,
                    topMargin=36,
                    bottomMargin=36,
                )
                styles = getSampleStyleSheet()
                styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8))
                styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=14, spaceAfter=6))
                styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], fontSize=12, spaceAfter=4))
                styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=11, leading=14))

                story = []

                def p(text: str, style_name: str):
                    safe = _sanitize_text(text)
                    story.append(Paragraph(escape(safe), styles[style_name]))

                for kind, payload in blocks:
                    if kind == "h1":
                        p(str(payload), "H1")
                    elif kind == "h2":
                        p(str(payload), "H2")
                    elif kind == "h3":
                        p(str(payload), "H3")
                    elif kind == "ul":
                        items = [ListItem(Paragraph(escape(_sanitize_text(it)), styles["Body"])) for it in payload]
                        story.append(ListFlowable(items, bulletType="bullet"))
                        story.append(Spacer(1, 6))
                    elif kind == "table":
                        rows = []
                        for r in payload:
                            rows.append([escape(_sanitize_text(c)) for c in r])
                        tbl = Table(rows, hAlign="LEFT")
                        tbl.setStyle(
                            TableStyle(
                                [
                                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                                    ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                                    ("FONT", (0, 1), (-1, -1), "Helvetica"),
                                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                                ]
                            )
                        )
                        story.append(tbl)
                        story.append(Spacer(1, 6))
                    else:
                        p(str(payload), "Body")
                        story.append(Spacer(1, 6))

                doc.build(story)
                return buffer.getvalue()

            pdf_bytes = _to_pdf_bytes(chosen.get("markdown_text", ""))
            st.download_button(
                label="Download Itinerary (PDF)",
                data=pdf_bytes,
                file_name="Itinerary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            if chosen.get("map_links"):
                st.subheader("🧭 Google Maps Links")
                for link in chosen["map_links"]:
                    st.markdown(f"- [{link['name']}]({link['url']})")
            if chosen.get("hotel_links"):
                st.subheader("🏨 Hotel Links")
                for link in chosen["hotel_links"]:
                    st.markdown(f"- [{link['name']}]({link['url']})")

        with col2:
            st.subheader("🗺️ Interactive Route Map")
            map_points = chosen.get('map_data', [])
            
            if map_points:
                start_lat = map_points[0]['lat']
                start_lon = map_points[0]['lon']
                m = folium.Map(location=[start_lat, start_lon], zoom_start=11)
                
                for idx, pt in enumerate(map_points):
                    color = 'red' if idx == 0 else 'blue'
                    icon = 'home' if idx == 0 else 'info-sign'
                    
                    folium.Marker(
                        [pt['lat'], pt['lon']],
                        popup=pt['name'],
                        tooltip=f"📍 {pt['name']}",
                        icon=folium.Icon(color=color, icon=icon)
                    ).add_to(m)
                    
                st_folium(m, width=400, height=450)
            else:
                st.info("No map coordinates generated for this trip.")
                
            st.subheader("✅ HITL Approval")
            if st.button("Authorize & Finalize Bookings", type="primary", use_container_width=True):
                st.balloons()
                st.success("Transaction Approved! Bookings secured.")

        if data.get("booking_ready") or data.get("booking_ready_variants"):
            st.divider()
            st.subheader("Booking-Ready Output")
            if data.get("booking_ready_variants") and chosen_title:
                match = next((v for v in data["booking_ready_variants"] if v.get("title") == chosen_title), None)
                st.markdown(match["booking_ready"] if match else data.get("booking_ready", ""))
            else:
                st.markdown(data.get("booking_ready", ""))
