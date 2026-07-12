"""
AI Air Travel Companion - Streamlit Frontend (app.py)
======================================================
Expedia Hackathon - Production-Grade Streamlit Application

Orchestrates Modules 1-5 into a professional, interactive dashboard.
Implements strict caching, session-state-driven anti-re-run execution,
and a 4-tab layout designed for hackathon judges.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Force CPU for Streamlit: CUDA tensors held in @st.cache_resource
# survive across Streamlit reruns but the CUDA context does not,
# causing "unknown error" on the second rerun. CPU tensors are
# always safe to share. Module 2 reads this env var at import time.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on the Python path so module imports resolve
# regardless of the cwd Streamlit happens to use.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module1_eda_cleaning import clean_flights, load_flights, load_users, build_airline_lookup, build_user_profiles
from module2_preference_engine import (
    PreferenceEmbedder,
    UserPreferenceStore,
    filter_candidates,
    score_and_rank,
    _DEVICE,
    _DTYPE,
)
from module3_orchestration import TravelCompanionPipeline, build_city_lookup
from module4_multi_city_router import MultiCityRouter
from module5_visualization import plot_interactive_route_map, plot_pareto_tradeoff, AIRPORT_COORDS

# ---------------------------------------------------------------------------
# Page Configuration — must be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Air Travel Companion",
    page_icon="\u2708",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-mode styling to match Module 5's Plotly theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Itinerary card */
    .itinerary-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #00e5ff33;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        box-shadow: 0 4px 20px rgba(0, 229, 255, 0.08);
    }
    .itinerary-card h3 { color: #00e5ff; margin-bottom: 8px; }
    .itinerary-card p  { color: #e0e0e0; line-height: 1.7; }

    /* Leg row */
    .leg-row {
        background: #0e1117;
        border-left: 3px solid #ff4081;
        padding: 12px 16px;
        margin: 6px 0;
        border-radius: 0 8px 8px 0;
        color: #e0e0e0;
    }

    /* KPI metric blocks */
    .kpi-block {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #00e5ff22;
    }
    .kpi-block .value { font-size: 28px; font-weight: 700; color: #00e5ff; }
    .kpi-block .label { font-size: 12px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }

    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        background: #ff408122;
        color: #ff4081;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
        border: 1px solid #ff408155;
    }

    /* Hide default Streamlit footer */
    footer { visibility: hidden; }

    /* Sidebar title styling */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #00e5ff;
        margin-bottom: 4px;
    }
    .sidebar-subtitle {
        font-size: 12px;
        color: #888;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================================== #
# 1.  DATA LOADING — @st.cache_data (runs once, result serialized)
# =========================================================================== #
@st.cache_data(show_spinner="Loading and cleaning 50,000 flight records...")
def load_and_clean_data():
    """Module 1 data pipeline — cached so it never re-runs on rerun."""
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("output")

    flights_clean_path = OUTPUT_DIR / "flights_clean.csv"
    profiles_path = OUTPUT_DIR / "user_profiles.json"

    # Prefer pre-computed outputs if they exist (from running module1 CLI)
    if flights_clean_path.exists() and profiles_path.exists():
        df_flights = pd.read_csv(
            flights_clean_path,
            parse_dates=["departure_utc", "arrival_utc"],
        )
        with open(profiles_path) as f:
            profiles = json.load(f)
    else:
        # Fallback: run Module 1 live
        df_flights_raw = load_flights(DATA_DIR / "flights_data.csv")
        df_users_raw = load_users(DATA_DIR / "user_data.csv")
        df_flights = clean_flights(df_flights_raw)
        airline_lookup = build_airline_lookup(df_flights)
        profiles = build_user_profiles(df_users_raw, airline_lookup)
        # Persist for future runs
        OUTPUT_DIR.mkdir(exist_ok=True)
        df_flights.to_csv(flights_clean_path, index=False)
        with open(profiles_path, "w") as f:
            json.dump(profiles, f, indent=2, default=str)

    return df_flights, profiles


# =========================================================================== #
# 2.  HEAVY RESOURCES — @st.cache_resource (runs once, kept in memory)
# =========================================================================== #
@st.cache_resource(show_spinner="Initializing AI engine (Transformer + FAISS + OR-Tools)...")
def build_ai_resources(_df_flights, _profiles_json_str: str):
    """
    Builds the deep-learning pipeline objects that live in GPU/CPU memory.
    _profiles_json_str: we pass profiles as a JSON string so Streamlit can
    hash it for cache invalidation (lists of dicts aren't hashable).
    """
    profiles = json.loads(_profiles_json_str)

    # Module 2: Transformer embedder + FAISS index
    embedder = PreferenceEmbedder(device="cpu")
    store = UserPreferenceStore(embedder)
    store.build(profiles)

    # Module 4: VRPTW router (pre-builds temporal graph)
    router = MultiCityRouter(_df_flights)

    # Module 3: End-to-end pipeline orchestrator
    pipeline = TravelCompanionPipeline(_df_flights, store, router=router, llm=None)

    # City lookup for sidebar UI
    city_lookup = build_city_lookup(_df_flights)

    return store, router, pipeline, city_lookup


# =========================================================================== #
# 3.  HELPER FUNCTIONS
# =========================================================================== #
def get_airport_display(code: str) -> str:
    """Format airport code + city for dropdown display."""
    info = AIRPORT_COORDS.get(code)
    if info:
        return f"{code} - {info['city']}"
    return code


def build_itinerary_df_from_legs(legs: list[dict]) -> pd.DataFrame:
    """Convert pipeline leg dicts into the DataFrame schema Module 5 expects."""
    rows = []
    for i, leg in enumerate(legs):
        if leg.get("status") != "OK":
            continue
        rows.append({
            "leg_order": i + 1,
            "origin": leg["origin"],
            "destination": leg["destination"],
            "date": str(leg.get("departure_utc", ""))[:10],
            "price": leg.get("price", 0),
            "duration_mins": leg.get("duration_hours", 0) * 60,
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_pareto_df_for_leg(df_flights: pd.DataFrame, origin: str, dest: str) -> pd.DataFrame:
    """Extract all candidate flights for a specific leg for the Pareto chart."""
    mask = (df_flights["origin"] == origin) & (df_flights["destination"] == dest)
    candidates = df_flights.loc[mask, [
        "flight_id", "airline_name", "price", "duration_minutes",
    ]].copy()
    candidates = candidates.rename(columns={"duration_minutes": "duration_mins"})
    return candidates.dropna(subset=["price", "duration_mins"]).reset_index(drop=True)


# =========================================================================== #
# 4.  MAIN APPLICATION
# =========================================================================== #
def main():
    # --- Load data (cached) ------------------------------------------------
    df_flights, profiles = load_and_clean_data()
    profiles_json_str = json.dumps(profiles, default=str)

    # --- Build AI resources (cached) ----------------------------------------
    store, router, pipeline, city_lookup = build_ai_resources(
        df_flights, profiles_json_str
    )

    # Reverse lookup: code -> city name
    code_to_city = {v: k.title() for k, v in city_lookup.items()}
    all_airports = sorted(df_flights["origin"].unique())

    # Build user-id -> profile lookup
    profile_map = {p["user_id"]: p for p in profiles}
    user_ids = sorted(profile_map.keys())

    # ===================================================================== #
    # SIDEBAR
    # ===================================================================== #
    with st.sidebar:
        st.markdown('<div class="sidebar-title">AI Air Travel Companion</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">Expedia Hackathon | VRPTW + Stochastic Beam Search</div>', unsafe_allow_html=True)
        st.divider()

        # --- User selection ---
        selected_user = st.selectbox(
            "Traveler Profile",
            user_ids,
            index=0,
            help="Each user has a unique AI-inferred preference profile mined from their travel history.",
        )
        profile = profile_map[selected_user]
        home_airport = profile["home_airport"]
        st.caption(f"Home Airport: **{get_airport_display(home_airport)}**")

        st.divider()

        # --- Destination selection ---
        dest_options = [a for a in all_airports if a != home_airport]
        default_dests = dest_options[:3] if len(dest_options) >= 3 else dest_options[:1]
        selected_destinations = st.multiselect(
            "Destinations",
            dest_options,
            default=default_dests,
            format_func=get_airport_display,
            help="Select one or more destination airports. Multi-city triggers VRPTW routing.",
        )

        st.divider()

        # --- Date selection ---
        min_date = df_flights["departure_utc"].min()
        max_date = df_flights["departure_utc"].max()
        # Provide defaults within the dataset range
        default_start = pd.Timestamp(min_date).date() if pd.notna(min_date) else date.today()
        selected_date = st.date_input(
            "Departure Date",
            value=default_start,
            min_value=default_start,
            max_value=pd.Timestamp(max_date).date() if pd.notna(max_date) else date.today() + timedelta(days=365),
            help="Earliest departure date for the first leg.",
        )
        flexibility = st.slider("Date Flexibility (days)", 1, 30, 7, help="How many days the solver can search around your departure date.")

        st.divider()

        # --- Weight sliders ---
        st.markdown("**Optimization Focus**")
        w_cost = st.slider("Cost Priority", 0.0, 1.0, float(profile["weights"].get("cost_weight", 0.3)), 0.05)
        w_time = st.slider("Speed Priority", 0.0, 1.0, float(profile["weights"].get("time_weight", 0.2)), 0.05)
        w_conv = st.slider("Convenience Priority", 0.0, 1.0, float(
            profile["weights"].get("stops_weight", 0.2)
            + profile["weights"].get("layover_weight", 0.1)
            + profile["weights"].get("reliability_weight", 0.05)
        ), 0.05)

        # Normalize
        w_total = w_cost + w_time + w_conv + 1e-10
        user_weights = {
            "cost_weight": round(w_cost / w_total, 3),
            "time_weight": round(w_time / w_total, 3),
            "stops_weight": round((w_conv / w_total) * 0.55, 3),
            "layover_weight": round((w_conv / w_total) * 0.35, 3),
            "reliability_weight": round((w_conv / w_total) * 0.10, 3),
        }

        st.divider()

        # --- Run button ---
        run_clicked = st.button(
            "Generate Optimized Itinerary",
            type="primary",
            use_container_width=True,
        )

    # ===================================================================== #
    # PIPELINE EXECUTION — only on button click
    # ===================================================================== #
    if run_clicked:
        if not selected_destinations:
            st.error("Please select at least one destination airport.")
            st.stop()

        with st.spinner("Running VRPTW solver + Stochastic Beam Search..."):
            try:
                is_multi_city = len(selected_destinations) >= 2
                start_date_str = selected_date.strftime("%Y-%m-%d")

                if is_multi_city:
                    # --- VRPTW multi-city path ---
                    route = router.solve_route(
                        home_airport,
                        selected_destinations,
                        user_preferences=user_weights,
                        start_date=start_date_str,
                        flexibility_days=flexibility,
                    )
                    itineraries = router.beam_search_itineraries(
                        route,
                        start_date=start_date_str,
                        user_preferences=user_weights,
                        beam_width=5,
                        mc_samples=100,
                    )
                    if not itineraries:
                        st.session_state["pipeline_result"] = {"error": "No feasible itinerary found."}
                    else:
                        best = itineraries[0]
                        # Get evidence from FAISS
                        evidence = store.evidence_for_query(selected_user, " ".join(selected_destinations), k=3)
                        # Build explanation
                        from module3_orchestration import explain_multi_city_template
                        explanation = explain_multi_city_template(
                            f"{home_airport} -> {' -> '.join(selected_destinations)}",
                            itineraries, evidence, profile,
                        )
                        # Distance matrix for engineering tab
                        dist_matrix = router.compute_distance_matrix(
                            [home_airport] + selected_destinations, user_weights
                        )
                        st.session_state["pipeline_result"] = {
                            "mode": "multi_city",
                            "vrptw_route": route,
                            "best_itinerary": {
                                "route": best.route,
                                "legs": best.legs,
                                "total_price": best.total_price,
                                "total_duration_hours": best.total_duration_hours,
                                "total_stops": best.total_stops,
                                "utility_score": best.utility_score,
                                "confidence": best.confidence,
                                "is_feasible": best.is_feasible,
                            },
                            "all_itineraries": [
                                {
                                    "route": it.route, "legs": it.legs,
                                    "total_price": it.total_price,
                                    "total_duration_hours": it.total_duration_hours,
                                    "total_stops": it.total_stops,
                                    "utility_score": it.utility_score,
                                    "confidence": it.confidence,
                                    "is_feasible": it.is_feasible,
                                }
                                for it in itineraries
                            ],
                            "explanation": explanation,
                            "evidence": evidence,
                            "profile": profile,
                            "user_weights": user_weights,
                            "distance_matrix": dist_matrix,
                            "matrix_cities": [home_airport] + selected_destinations,
                        }
                else:
                    # --- Single-destination path ---
                    dest = selected_destinations[0]
                    candidates = filter_candidates(
                        df_flights,
                        origin=home_airport,
                        destinations=dest,
                    )
                    if candidates.empty:
                        st.session_state["pipeline_result"] = {
                            "error": f"No flights found from {home_airport} to {dest}."
                        }
                    else:
                        # Override profile weights with slider values
                        modified_profile = dict(profile)
                        modified_profile["weights"] = user_weights
                        result = score_and_rank(candidates, modified_profile, top_k=5)
                        evidence = store.evidence_for_query(selected_user, dest, k=3)

                        from module3_orchestration import explain_template
                        top_pick = result["top_k"][0] if result["top_k"] else None
                        explanation = ""
                        if top_pick:
                            explanation = explain_template(
                                f"{home_airport} -> {dest}",
                                top_pick, result["pareto_frontier"], evidence, modified_profile,
                            )
                        st.session_state["pipeline_result"] = {
                            "mode": "single",
                            "origin": home_airport,
                            "destination": dest,
                            "top_k": result["top_k"],
                            "pareto_frontier": result["pareto_frontier"],
                            "learned_weights": result["learned_weights"],
                            "explanation": explanation,
                            "evidence": evidence,
                            "profile": modified_profile,
                            "user_weights": user_weights,
                            "all_candidates": candidates,
                        }

            except Exception as e:
                st.session_state["pipeline_result"] = {"error": str(e)}

    # ===================================================================== #
    # MAIN WORKSPACE — render from session_state
    # ===================================================================== #
    result = st.session_state.get("pipeline_result")

    if result is None:
        # --- Welcome state ---
        st.markdown("## AI Air Travel Companion")
        st.markdown(
            "Configure your trip in the sidebar and click **Generate Optimized Itinerary** "
            "to run the VRPTW solver with stochastic beam search."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="kpi-block">
                <div class="value">50K</div>
                <div class="label">Flight Records</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="kpi-block">
                <div class="value">{len(profiles)}</div>
                <div class="label">User Profiles</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="kpi-block">
                <div class="value">{len(all_airports)}</div>
                <div class="label">Airports</div>
            </div>""", unsafe_allow_html=True)

        st.info(
            "**Tech Stack**: PyTorch + FAISS (Adaptive IVF+HNSW) + "
            "OR-Tools VRPTW + Stochastic Beam Search + Monte Carlo Confidence",
        )
        return

    if "error" in result:
        st.error(f"Pipeline Error: {result['error']}")
        return

    # ===================================================================== #
    # TABS
    # ===================================================================== #
    tab1, tab2, tab3, tab4 = st.tabs([
        "Your AI Companion Itinerary",
        "Interactive Routing Map",
        "Pareto Trade-Off Analysis",
        "Engineering & Graph Logs",
    ])

    # ------------------------------------------------------------------ #
    # TAB 1 — AI Explanation + Itinerary Cards
    # ------------------------------------------------------------------ #
    with tab1:
        st.markdown("### AI-Generated Travel Recommendation")

        if result["mode"] == "multi_city":
            best = result["best_itinerary"]

            # KPI row
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.markdown(f"""<div class="kpi-block">
                    <div class="value">${best['total_price']:,.0f}</div>
                    <div class="label">Total Price</div>
                </div>""", unsafe_allow_html=True)
            with k2:
                st.markdown(f"""<div class="kpi-block">
                    <div class="value">{best['total_duration_hours']:.1f}h</div>
                    <div class="label">Total Duration</div>
                </div>""", unsafe_allow_html=True)
            with k3:
                st.markdown(f"""<div class="kpi-block">
                    <div class="value">{best['total_stops']}</div>
                    <div class="label">Total Stops</div>
                </div>""", unsafe_allow_html=True)
            with k4:
                st.markdown(f"""<div class="kpi-block">
                    <div class="value">{best['confidence']:.0%}</div>
                    <div class="label">MC Confidence</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("")

            # Explanation card
            st.markdown(f"""<div class="itinerary-card">
                <h3>Route: {' -> '.join(best['route'])}</h3>
                <p>{result['explanation'].replace(chr(10), '<br>')}</p>
            </div>""", unsafe_allow_html=True)

            # Leg detail cards
            st.markdown("#### Flight Legs")
            for i, leg in enumerate(best["legs"]):
                if leg.get("status") == "OK":
                    st.markdown(f"""<div class="leg-row">
                        <strong>Leg {i+1}:</strong> {leg['origin']} -> {leg['destination']}
                        &nbsp;|&nbsp; {leg.get('airline_name', 'N/A')}
                        &nbsp;|&nbsp; <strong>${leg['price']:.0f}</strong>
                        &nbsp;|&nbsp; {leg['duration_hours']:.1f}h
                        &nbsp;|&nbsp; {leg['stops']} stop(s)
                        {f"&nbsp;|&nbsp; Date gap: {leg['date_gap_days']}d" if leg.get('date_gap_days', 0) > 1 else ""}
                    </div>""", unsafe_allow_html=True)
                else:
                    st.warning(f"Leg {i+1}: {leg['origin']} -> {leg['destination']} -- INFEASIBLE ({leg.get('reason', 'unknown')})")

            # Alternative itineraries
            all_itins = result.get("all_itineraries", [])
            if len(all_itins) > 1:
                st.markdown("#### Beam Search Alternatives")
                alt_data = []
                for idx, it in enumerate(all_itins):
                    alt_data.append({
                        "Rank": idx + 1,
                        "Route": " -> ".join(it["route"]),
                        "Price ($)": f"${it['total_price']:,.0f}",
                        "Duration (h)": f"{it['total_duration_hours']:.1f}",
                        "Stops": it["total_stops"],
                        "Utility": f"{it['utility_score']:.3f}",
                        "Confidence": f"{it['confidence']:.0%}",
                        "Feasible": "Yes" if it["is_feasible"] else "No",
                    })
                st.dataframe(pd.DataFrame(alt_data), use_container_width=True, hide_index=True)

        else:
            # Single-destination mode
            top_k = result.get("top_k", [])
            if top_k:
                top = top_k[0]
                k1, k2, k3 = st.columns(3)
                with k1:
                    st.markdown(f"""<div class="kpi-block">
                        <div class="value">${top['price']:,.0f}</div>
                        <div class="label">Best Price</div>
                    </div>""", unsafe_allow_html=True)
                with k2:
                    st.markdown(f"""<div class="kpi-block">
                        <div class="value">{top['duration_hours']:.1f}h</div>
                        <div class="label">Duration</div>
                    </div>""", unsafe_allow_html=True)
                with k3:
                    st.markdown(f"""<div class="kpi-block">
                        <div class="value">{top['stops']}</div>
                        <div class="label">Stops</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="itinerary-card">
                <h3>{result['origin']} -> {result['destination']}</h3>
                <p>{result['explanation'].replace(chr(10), '<br>')}</p>
            </div>""", unsafe_allow_html=True)

            if len(top_k) > 1:
                st.markdown("#### Top Scored Flights")
                rows = []
                for f in top_k:
                    rows.append({
                        "Airline": f.get("airline_name", "N/A"),
                        "Price ($)": f"${f['price']:,.0f}",
                        "Duration (h)": f"{f['duration_hours']:.1f}",
                        "Stops": f["stops"],
                        "Utility": f"{f['utility_score']:.3f}",
                        "Pareto": "Yes" if f.get("is_pareto_optimal") else "No",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------ #
    # TAB 2 — Geospatial Route Map
    # ------------------------------------------------------------------ #
    with tab2:
        st.markdown("### Interactive Route Map")
        try:
            if result["mode"] == "multi_city":
                best = result["best_itinerary"]
                itin_df = build_itinerary_df_from_legs(best["legs"])
            else:
                top_k = result.get("top_k", [])
                if top_k:
                    top = top_k[0]
                    itin_df = pd.DataFrame([{
                        "leg_order": 1,
                        "origin": result["origin"],
                        "destination": result["destination"],
                        "date": str(top.get("departure_utc", ""))[:10],
                        "price": top["price"],
                        "duration_mins": top["duration_hours"] * 60,
                    }])
                else:
                    itin_df = pd.DataFrame()

            if itin_df.empty:
                st.warning("No feasible legs to plot on the map.")
            else:
                fig_map = plot_interactive_route_map(itin_df)
                st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": True})
        except Exception as e:
            st.error(f"Map rendering error: {e}")

    # ------------------------------------------------------------------ #
    # TAB 3 — Pareto Trade-Off
    # ------------------------------------------------------------------ #
    with tab3:
        st.markdown("### Pareto Trade-Off Analysis (Price vs. Duration)")
        try:
            if result["mode"] == "multi_city":
                best = result["best_itinerary"]
                feasible_legs = [l for l in best["legs"] if l.get("status") == "OK"]
                if not feasible_legs:
                    st.warning("No feasible legs for Pareto analysis.")
                else:
                    selected_leg_idx = st.selectbox(
                        "Select a leg to analyze",
                        range(len(feasible_legs)),
                        format_func=lambda i: f"Leg {i+1}: {feasible_legs[i]['origin']} -> {feasible_legs[i]['destination']}",
                    )
                    leg = feasible_legs[selected_leg_idx]
                    pareto_df = build_pareto_df_for_leg(df_flights, leg["origin"], leg["destination"])
                    if pareto_df.empty:
                        st.warning(f"No flight candidates found for {leg['origin']} -> {leg['destination']}.")
                    else:
                        selected_id = leg.get("flight_id", "")
                        fig_pareto = plot_pareto_tradeoff(pareto_df, selected_id)
                        st.plotly_chart(fig_pareto, use_container_width=True)
                        st.caption(f"Showing {len(pareto_df)} flights for {leg['origin']} -> {leg['destination']}. "
                                   f"The AI-selected flight (star) is '{selected_id}'.")
            else:
                # Single-destination: use all candidates
                candidates = result.get("all_candidates")
                top_k = result.get("top_k", [])
                if candidates is not None and not candidates.empty and top_k:
                    pareto_input = candidates[["flight_id", "airline_name", "price", "duration_minutes"]].copy()
                    pareto_input = pareto_input.rename(columns={"duration_minutes": "duration_mins"})
                    selected_id = top_k[0].get("flight_id", "")
                    fig_pareto = plot_pareto_tradeoff(pareto_input, selected_id)
                    st.plotly_chart(fig_pareto, use_container_width=True)
                    st.caption(f"Showing {len(pareto_input)} candidate flights. "
                               f"AI recommendation: '{selected_id}'.")
                else:
                    st.warning("No candidate data available for Pareto analysis.")
        except Exception as e:
            st.error(f"Pareto chart error: {e}")

    # ------------------------------------------------------------------ #
    # TAB 4 — Engineering & Graph Logs
    # ------------------------------------------------------------------ #
    with tab4:
        st.markdown("### Engineering & Technical Deep-Dive")
        st.caption("Raw pipeline internals for technical judges.")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### User Profile (Extracted Constraints)")
            display_profile = {
                "user_id": result["profile"]["user_id"],
                "home_airport": result["profile"]["home_airport"],
                "preferred_cabin": result["profile"].get("preferred_cabin"),
                "price_sensitivity": result["profile"].get("price_sensitivity"),
                "direct_preference": result["profile"].get("direct_preference"),
                "max_layover_minutes": result["profile"].get("max_layover_minutes"),
                "date_flexibility_days": result["profile"].get("date_flexibility_days"),
                "trip_purpose": result["profile"].get("trip_purpose"),
            }
            st.json(display_profile)

            st.markdown("#### AI-Inferred Weights (5-Axis)")
            st.json(result["user_weights"])

            st.markdown("#### FAISS-Retrieved Evidence")
            for i, ev in enumerate(result.get("evidence", [])):
                st.code(f"[{i+1}] {ev}", language=None)

        with col_right:
            st.markdown("#### Compute Configuration")
            st.json({
                "torch_device": str(_DEVICE),
                "torch_dtype": str(_DTYPE),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
                "n_flights": len(df_flights),
                "n_profiles": len(profiles),
                "n_airports": len(all_airports),
            })

            if result["mode"] == "multi_city":
                st.markdown("#### OR-Tools VRPTW Route")
                st.code(" -> ".join(result.get("vrptw_route", [])), language=None)

                st.markdown("#### Integer Distance Matrix (OR-Tools Input)")
                matrix = result.get("distance_matrix")
                cities = result.get("matrix_cities", [])
                if matrix:
                    matrix_df = pd.DataFrame(matrix, index=cities, columns=cities)
                    st.dataframe(matrix_df, use_container_width=True)
                else:
                    st.info("Distance matrix not available.")

                st.markdown("#### Beam Search Raw Output")
                all_itins = result.get("all_itineraries", [])
                for i, it in enumerate(all_itins):
                    with st.expander(f"Itinerary #{i+1} (utility={it['utility_score']:.3f})"):
                        st.json({
                            "route": it["route"],
                            "total_price": it["total_price"],
                            "total_duration_hours": it["total_duration_hours"],
                            "total_stops": it["total_stops"],
                            "confidence": it["confidence"],
                            "is_feasible": it["is_feasible"],
                            "n_legs": len(it["legs"]),
                        })
            else:
                st.markdown("#### Learned Utility Weights (PyTorch)")
                st.json(result.get("learned_weights", {}))

                st.markdown("#### Pareto Frontier Summary")
                pareto = result.get("pareto_frontier", [])
                st.metric("Pareto-optimal flights", len(pareto))
                if pareto:
                    prices = [f["price"] for f in pareto]
                    st.caption(f"Price range: ${min(prices):,.0f} - ${max(prices):,.0f}")


if __name__ == "__main__":
    main()
