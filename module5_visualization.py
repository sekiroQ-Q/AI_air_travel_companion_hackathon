"""
Module 5 - Visualization Engine (Streamlit Ready)
=================================================
AI Air Travel Companion (hackathon prototype)

Generates highly interactive, visually stunning Plotly charts for the
frontend. Designed to be rendered in Streamlit via st.plotly_chart().

1. plot_interactive_route_map: A geospatial visualization of the
   multi-city itinerary, sequencing the flight paths.
2. plot_pareto_tradeoff: An analytical scatter plot proving the
   selected flight's position on the cost-time Pareto frontier.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# A demo dictionary of global airport coordinates for mapping.
# In a full production app, this would be loaded from a database or a file (e.g., ourairports.com).
AIRPORT_COORDS = {
    # India
    "DEL": {"lat": 28.5562, "lon": 77.1000, "city": "New Delhi"},
    "BOM": {"lat": 19.0896, "lon": 72.8656, "city": "Mumbai"},
    "BLR": {"lat": 13.1986, "lon": 77.7066, "city": "Bangalore"},
    "HYD": {"lat": 17.2403, "lon": 78.4294, "city": "Hyderabad"},
    "MAA": {"lat": 12.9941, "lon": 80.1709, "city": "Chennai"},
    # Europe
    "LHR": {"lat": 51.4700, "lon": -0.4543, "city": "London"},
    "CDG": {"lat": 49.0097, "lon": 2.5479, "city": "Paris"},
    "FCO": {"lat": 41.7999, "lon": 12.2462, "city": "Rome"},
    "AMS": {"lat": 52.3105, "lon": 4.7683, "city": "Amsterdam"},
    "FRA": {"lat": 50.0379, "lon": 8.5622, "city": "Frankfurt"},
    "BCN": {"lat": 41.2974, "lon": 2.0833, "city": "Barcelona"},
    # Americas
    "JFK": {"lat": 40.6413, "lon": -73.7781, "city": "New York"},
    "SFO": {"lat": 37.6213, "lon": -122.3790, "city": "San Francisco"},
    "MEX": {"lat": 19.4361, "lon": -99.0719, "city": "Mexico City"},
    # Africa & Middle East
    "CPT": {"lat": -33.9715, "lon": 18.6021, "city": "Cape Town"},
    "DXB": {"lat": 25.2532, "lon": 55.3657, "city": "Dubai"},
    "DOH": {"lat": 25.2731, "lon": 51.6080, "city": "Doha"},
    # Asia & Oceania
    "SIN": {"lat": 1.3644, "lon": 103.9915, "city": "Singapore"},
    "NRT": {"lat": 35.7647, "lon": 140.3863, "city": "Tokyo"},
    "SYD": {"lat": -33.9461, "lon": 151.1772, "city": "Sydney"},
    "AKL": {"lat": -37.0082, "lon": 174.7850, "city": "Auckland"},
    "BKK": {"lat": 13.6900, "lon": 100.7501, "city": "Bangkok"}
}

def plot_interactive_route_map(itinerary_df: pd.DataFrame) -> go.Figure:
    """
    Plots a sequenced, animated-style line showing the multi-city journey
    on a geographic map.

    Expected itinerary_df columns: ['leg_order', 'origin', 'destination', 'date', 'price', 'duration_mins']

    Returns a plotly Figure object that can be:
      - Rendered in Streamlit via st.plotly_chart(fig)
      - Saved to disk via fig.write_html("path.html")
    """
    fig = go.Figure()

    # Sort to ensure we draw the path in the correct chronological sequence
    df = itinerary_df.copy()
    if 'leg_order' in df.columns:
        df = df.sort_values('leg_order').reset_index(drop=True)

    lats = []
    lons = []
    hover_texts = []
    is_first = True  # track first leg explicitly (idx may not be 0 after sort)

    # Plot individual flight arcs
    for i in range(len(df)):
        row = df.iloc[i]
        origin = row['origin']
        dest = row['destination']
        leg_num = row.get('leg_order', i + 1)

        # Fallback to (0,0) if coordinate isn't in our demo dictionary
        o_coords = AIRPORT_COORDS.get(origin, {"lat": 0, "lon": 0, "city": origin})
        d_coords = AIRPORT_COORDS.get(dest, {"lat": 0, "lon": 0, "city": dest})

        # Add the line between origin and destination
        fig.add_trace(go.Scattergeo(
            lat=[o_coords["lat"], d_coords["lat"]],
            lon=[o_coords["lon"], d_coords["lon"]],
            mode="lines",
            line=dict(width=3, color="#00e5ff"),  # Vibrant neon cyan for the path
            opacity=0.8,
            name=f"Leg {leg_num}",
            hoverinfo="text",
            text=f"Leg {leg_num}: {origin} -> {dest}<br>"
                 f"Price: ${row.get('price', 0):.0f}<br>"
                 f"Duration: {row.get('duration_mins', 0) / 60:.1f}h",
        ))

        # Keep track of nodes for the scatter points
        if is_first:
            lats.append(o_coords["lat"])
            lons.append(o_coords["lon"])
            hover_texts.append(f"Start: {origin} ({o_coords['city']})")
            is_first = False

        lats.append(d_coords["lat"])
        lons.append(d_coords["lon"])
        hover_texts.append(f"Stop {leg_num}: {dest} ({d_coords['city']})")

    # Plot the airports as distinct nodes
    # Extract short labels (airport code) from hover_texts for on-map display
    labels = []
    for txt in hover_texts:
        # "Start: JFK (New York)" -> "JFK" or "Stop 1: LHR (London)" -> "LHR"
        after_colon = txt.split(":", 1)[1] if ":" in txt else txt
        code = after_colon.split("(")[0].strip()
        labels.append(code)

    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        mode="markers+text",
        marker=dict(
            size=12,
            color="#ff4081",  # Vibrant neon pink for nodes
            line=dict(width=2, color="white"),
            symbol="circle"
        ),
        text=labels,
        textposition="bottom center",
        textfont=dict(color="white", size=11),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Airports"
    ))

    # Polish the map styling for a premium 'Hackathon' aesthetic
    fig.update_layout(
        title=dict(
            text="Optimal Multi-City Flight Path",
            font=dict(size=24, color="white"),
            x=0.5
        ),
        geo=dict(
            showland=True,
            landcolor="#1e1e24",       # Sleek dark grey land
            showocean=True,
            oceancolor="#0e1117",      # Matches Streamlit's dark mode background
            showcountries=True,
            countrycolor="#444444",
            showlakes=False,
            projection_type="equirectangular",
            coastlinecolor="#444444",
            bgcolor="#0e1117",
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=False,
        hoverlabel=dict(bgcolor="black", font_size=14, font_family="Arial")
    )
    return fig


def plot_pareto_tradeoff(all_flights_df: pd.DataFrame, selected_flight_id: str) -> go.Figure:
    """
    An analytical scatter plot showing Duration vs. Price for all available
    flights on a specific route, highlighting the Pareto frontier and the
    flight chosen by the optimizer.

    Expected all_flights_df columns: ['flight_id', 'airline_name', 'price', 'duration_mins']

    Returns a plotly Figure object.
    """
    # Ensure data is clean
    df = all_flights_df.dropna(subset=['price', 'duration_mins']).copy()
    df['duration_hours'] = df['duration_mins'] / 60.0

    # Calculate simple Pareto frontier mask (Price & Duration minimization)
    # Sort by price ascending, then find points where duration is strictly
    # less than ALL cheaper flights seen so far (i.e. a step-down in duration
    # as price rises). This is the classic 2-objective sweep-line Pareto.
    df = df.sort_values('price').reset_index(drop=True)
    pareto_mask = []
    min_dur_so_far = float('inf')

    for i in range(len(df)):
        dur = df.iloc[i]['duration_hours']
        if dur < min_dur_so_far:
            pareto_mask.append(True)
            min_dur_so_far = dur
        else:
            pareto_mask.append(False)

    df['is_pareto'] = pareto_mask

    # Separate the data for coloring
    pareto_df = df[df['is_pareto']].sort_values('duration_hours')
    non_pareto_df = df[~df['is_pareto']]

    selected_flight = df[df['flight_id'] == selected_flight_id]

    fig = go.Figure()

    # 1. Non-Pareto flights (background)
    fig.add_trace(go.Scatter(
        x=non_pareto_df['duration_hours'],
        y=non_pareto_df['price'],
        mode='markers',
        name='Other Alternatives',
        marker=dict(color='#555555', size=8, opacity=0.6),
        text=non_pareto_df['airline_name'],
        hovertemplate="<b>%{text}</b><br>Time: %{x:.1f}h<br>Price: $%{y:.0f}<extra></extra>"
    ))

    # 2. Pareto Frontier Line (connects the optimal points)
    fig.add_trace(go.Scatter(
        x=pareto_df['duration_hours'],
        y=pareto_df['price'],
        mode='lines+markers',
        name='Pareto Frontier (Optimal)',
        line=dict(color='#00e5ff', width=2, dash='dash'),
        marker=dict(color='#00e5ff', size=10, line=dict(color='white', width=1)),
        text=pareto_df['airline_name'],
        hovertemplate="<b>%{text} (Optimal)</b><br>Time: %{x:.1f}h<br>Price: $%{y:.0f}<extra></extra>"
    ))

    # 3. Highlight the specifically selected flight
    if not selected_flight.empty:
        fig.add_trace(go.Scatter(
            x=selected_flight['duration_hours'],
            y=selected_flight['price'],
            mode='markers+text',
            name='AI Selected Flight',
            marker=dict(
                color='#ff4081',
                size=18,
                symbol='star',
                line=dict(color='white', width=2)
            ),
            text=["  AI Pick"],
            textposition="middle right",
            textfont=dict(color="#ff4081", size=14),
            hovertemplate="<b>AI Recommendation</b><br>Time: %{x:.1f}h<br>Price: $%{y:.0f}<extra></extra>"
        ))

    # Premium Aesthetic Formatting
    fig.update_layout(
        title=dict(
            text="Cost vs. Time Trade-Off Analysis",
            font=dict(size=24, color="white")
        ),
        xaxis=dict(
            title="Total Duration (Hours)",
            gridcolor="#333333",
            zerolinecolor="#333333",
            color="white",
            showgrid=True
        ),
        yaxis=dict(
            title="Total Price (USD)",
            gridcolor="#333333",
            zerolinecolor="#333333",
            color="white",
            showgrid=True,
            tickprefix="$"
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white")
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="closest"
    )

    # Highlight the "ideal" quadrant (bottom-left)
    fig.add_annotation(
        x=df['duration_hours'].min() * 0.95,
        y=df['price'].min() * 0.95,
        text="Fast & Cheap",
        showarrow=False,
        font=dict(color="#00e5ff", size=12),
        bgcolor="rgba(0, 229, 255, 0.1)",
        bordercolor="#00e5ff",
        borderwidth=1,
        borderpad=4
    )

    return fig


# =========================================================================== #
# Demo / Smoke Test — saves interactive HTML to output/
# =========================================================================== #
if __name__ == "__main__":
    from pathlib import Path

    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # ----- Test 1: Interactive Route Map -----
    print("Building geospatial route map...")
    dummy_itinerary = pd.DataFrame([
        {"leg_order": 1, "origin": "JFK", "destination": "LHR", "date": "2026-07-20", "price": 450, "duration_mins": 420},
        {"leg_order": 2, "origin": "LHR", "destination": "CDG", "date": "2026-07-23", "price": 85,  "duration_mins": 75},
        {"leg_order": 3, "origin": "CDG", "destination": "FCO", "date": "2026-07-27", "price": 120, "duration_mins": 125},
        {"leg_order": 4, "origin": "FCO", "destination": "DEL", "date": "2026-07-30", "price": 600, "duration_mins": 450},
    ])

    fig_map = plot_interactive_route_map(dummy_itinerary)
    map_path = OUTPUT_DIR / "route_map.html"
    fig_map.write_html(str(map_path))
    print(f"  -> saved interactive map to {map_path}")

    # ----- Test 2: Pareto Trade-Off Chart -----
    print("Building Pareto trade-off scatter plot...")
    rng = np.random.default_rng(42)
    prices = rng.normal(500, 150, 50).clip(150, 1000)
    durations = (1000 - prices + rng.normal(0, 100, 50)).clip(120, 900)

    dummy_flights = pd.DataFrame({
        "flight_id": [f"F{i:03d}" for i in range(50)],
        "airline_name": rng.choice(["Delta", "Air India", "Emirates", "Lufthansa"], 50),
        "price": prices,
        "duration_mins": durations,
    })

    # Force one flight to be explicitly excellent (on the Pareto frontier)
    dummy_flights.loc[0, "price"] = 350
    dummy_flights.loc[0, "duration_mins"] = 300
    dummy_flights.loc[0, "flight_id"] = "AI_PICK_001"

    fig_pareto = plot_pareto_tradeoff(dummy_flights, "AI_PICK_001")
    pareto_path = OUTPUT_DIR / "pareto_tradeoff.html"
    fig_pareto.write_html(str(pareto_path))
    print(f"  -> saved Pareto chart to {pareto_path}")

    print("\nDone! Open the HTML files in a browser to view the interactive charts.")
