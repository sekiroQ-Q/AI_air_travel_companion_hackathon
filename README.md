# AI Air Travel Companion — Hackathon Prototype

A high-performance, modular pipeline for the Expedia Hackathon, demonstrating deep technical rigor and innovation in multi-city travel optimization. Ships with a complete **Streamlit dashboard** for live demo.

## Project Structure

```
app.py                       -> Streamlit frontend (the demo entry point)
module1_eda_cleaning.py      -> output/flights_clean.csv, output/user_profiles.json, output/eda_plots/
module2_preference_engine.py -> Transformer embeddings + FAISS store + PyTorch multi-objective optimizer
module3_orchestration.py     -> End-to-end pipeline orchestrator (query -> optimize -> explain)
module4_multi_city_router.py -> VRPTW (OR-Tools) + Stochastic Beam Search for multi-city routing
module5_visualization.py     -> Plotly charts (Geospatial Route Maps & Pareto Trade-Off Scatter)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            torch faiss-cpu sentence-transformers \
            transformers langchain langchain-community \
            ortools plotly streamlit
```

### 2. Prepare Data

```
data/flights_data.csv
data/user_data.csv
```

### 3. Run the Pipeline (one-time data prep)

```bash
python module1_eda_cleaning.py
```

This generates `output/flights_clean.csv` and `output/user_profiles.json`. Modules 2-5 consume these outputs.

### 4. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. That's it — the app handles everything else.

> **Note**: You can also run each module individually for debugging:
> ```bash
> python module2_preference_engine.py   # FAISS + optimizer smoke test
> python module4_multi_city_router.py   # VRPTW + beam search demo (dummy + real data)
> python module5_visualization.py       # Saves interactive HTML charts to output/
> python module3_orchestration.py       # Full end-to-end pipeline (CLI mode)
> ```

## Streamlit Dashboard (`app.py`)

The frontend is a production-grade Streamlit app with strict caching and state management:

### Caching Strategy
| Decorator | What it caches | Runs when |
|---|---|---|
| `@st.cache_data` | 50K flight records + 50 user profiles | Once on first load |
| `@st.cache_resource` | Transformer model, FAISS index, OR-Tools VRPTW router | Once on first load |

### Anti-Re-run Execution
All heavy computation (VRPTW solving, beam search, PyTorch scoring) is gated behind the **"Generate Optimized Itinerary"** button. Results are stored in `st.session_state`. Switching tabs, adjusting sliders, or interacting with charts **never** re-triggers the optimization backend.

### Layout

**Sidebar Controls:**
- Traveler Profile dropdown (50 AI-profiled users)
- Multi-select Destinations (auto-routes to VRPTW when >= 2 cities selected)
- Departure Date picker + Flexibility slider
- 3 Optimization Focus sliders (Cost / Speed / Convenience) — auto-normalized to the 5-axis weight vector

**Main Workspace (4 Tabs):**

| Tab | Content |
|---|---|
| **Your AI Companion Itinerary** | KPI cards (price, duration, stops, MC confidence), NL explanation, per-leg detail cards, beam search alternatives table |
| **Interactive Routing Map** | Plotly `Scattergeo` global flight path map with neon-styled dark theme |
| **Pareto Trade-Off Analysis** | Interactive scatter plot proving the AI pick sits on the cost-time Pareto frontier. Leg selector for multi-city |
| **Engineering & Graph Logs** | Raw JSON of user profile, 5-axis weights, FAISS evidence, OR-Tools integer distance matrix, compute config (CUDA/dtype/numpy) |

## Core Architecture & SOTA Upgrades

### 1. Vectorized Data Engineering (Module 1)
- Replaced `df.iterrows()` with **Vectorized Batch Processing** (`Series.apply` + pre-compiled regex).
- Pre-computes stochastic price distributions (`route_price_mean`, `route_price_std`, `route_price_median`) for Module 4's Monte Carlo sampling.
- Adds `flight_date` column for temporal edge construction.

### 2. Adaptive GPU Preference Engine (Module 2)
- **Adaptive FAISS**: `IndexFlatIP` (exact) for n < 1,000; `IVF+HNSW` (approximate, O(log n)) for larger datasets.
- **Vectorized Pareto Dominance**: NumPy broadcasting replaces O(n^2) Python loops — ~50x constant-factor speedup.
- **GPU Inference**: CUDA + FP16 Automatic Mixed Precision via `_make_tensor()` factory, with explicit VRAM cleanup after every pass.

### 3. VRPTW + Stochastic Beam Search (Module 4)
The technical centerpiece:
- **Temporal Flight Graph**: Each (origin, dest, date) is a discrete edge — not an aggregate. Enables date-aware routing.
- **VRPTW via OR-Tools**: Jointly optimizes city ordering AND temporal feasibility with time windows, eliminating post-hoc date-gap hacks.
- **Stochastic Beam Search**: Explores `beam_width` partial itineraries in parallel per leg. Each complete itinerary's price robustness is scored via **Monte Carlo sampling** against per-route variance distributions.

### 4. Interactive Visualizations (Module 5)
- Premium dark-mode **Plotly** charts matching Streamlit's aesthetic.
- **Geospatial Route Map**: Sequenced flight paths on a global map with airport nodes.
- **Pareto Trade-Off Chart**: All candidate flights plotted as Duration vs. Price, with the Pareto frontier line and AI-selected flight highlighted as a star.
- Outputs saved as interactive HTML files (`output/route_map.html`, `output/pareto_tradeoff.html`) when run standalone.

### 5. Pipeline Orchestration (Module 3)
- Automatically detects single-destination vs. multi-city queries.
- Single-destination: Module 2's constrained multi-objective optimizer.
- Multi-city (>= 2 destinations): Module 4's VRPTW + beam search.
- Evidence-cited natural language explanations with trade-off articulation.

## Offline Fallback Behavior

Built for demo-day resilience — **no API outage can crash the demo**:
- **Embeddings**: Tries `sentence-transformers/all-MiniLM-L6-v2` first. Falls back to local TF-IDF + SVD if HuggingFace is unreachable.
- **Query Parsing & Explanation**: Deterministic rule-based parser + template explainer as default. LLM can be wired in via the `llm` parameter.

## Tech Stack

| Component | Technology |
|---|---|
| Data Engineering | Pandas (vectorized), NumPy 2.2.3 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim) |
| Vector Search | FAISS (Adaptive: Flat / IVF+HNSW) |
| Multi-Objective Scoring | PyTorch (CUDA FP16, 5-axis utility) |
| Route Optimization | Google OR-Tools (VRPTW + Guided Local Search) |
| Flight Resolution | Stochastic Beam Search + Monte Carlo Confidence |
| Visualization | Plotly (Scattergeo + Scatter) |
| Frontend | Streamlit (cached resources, session-state gating) |
| Explanation | LangChain-compatible (rule-based default, LLM optional) |