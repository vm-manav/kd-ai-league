import streamlit as st
import folium
from streamlit_folium import st_folium
from agent import run_travel_agent

st.set_page_config(page_title="Swarm Travel Planner", layout="wide", page_icon="🌍")

st.title("🌍 Autonomous Travel Swarm")
st.markdown("Powered by LangGraph Parallel Agents (Booking, Weather, Local Expert)")

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = "user_123"
if 'trip_data' not in st.session_state:
    st.session_state.trip_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Chat Interface
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("E.g., Plan a luxury trip to Jaipur next week")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Agents are executing in parallel..."):
        result = run_travel_agent(user_input, thread_id=st.session_state.thread_id)
        st.session_state.trip_data = result

        if result.get("missing_info_question"):
            msg = result["missing_info_question"]
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"):
                st.markdown(msg)
            result["missing_info_question"] = "" 
            st.stop()

    if result.get("final_itinerary"):
        st.session_state.chat_history.append({"role": "assistant", "content": "Trip finalized! See details below."})
        with st.chat_message("assistant"):
            st.markdown("Trip finalized! Review your package below.")

# Render Dashboard
if st.session_state.trip_data and st.session_state.trip_data.get("final_itinerary"):
    data = st.session_state.trip_data
    
    st.divider()
    
    if "WARNING" in data.get('weather_alert', ''):
        st.error(data['weather_alert'])
    else:
        st.success(data.get('weather_alert', 'Weather checked successfully.'))

    col1, col2 = st.columns([2, 1.5])
    
    with col1:
        st.subheader("🗓️ Execution Package")
        st.markdown(data['final_itinerary'])
        
        st.download_button(
            label="📥 Download Itinerary (.md)",
            data=data['final_itinerary'],
            file_name="Itinerary.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        st.subheader("🗺️ Interactive Route Map")
        map_points = data.get('map_data', [])
        
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