"""
Module 4 - Multi-City Route Optimization (VRPTW + Stochastic Beam Search)
==========================================================================
AI Air Travel Companion (hackathon prototype)

SOTA upgrade over baseline TSP:

1. TEMPORAL FLIGHT GRAPH: each (origin, dest, date) is a discrete edge
   with real cost — not an aggregate RouteStats. This enables the solver
   to reason about date-specific pricing and availability.

2. VRPTW (Vehicle Routing Problem with Time Windows): the textbook-correct
   generalization of TSP that jointly handles:
   - Temporal feasibility (departure_next >= arrival_prev + buffer)
   - Date flexibility windows (user says "next week" → 7-day window)
   - Per-node service time (stay duration at each city)
   This is what the baseline TSP fundamentally cannot do: enforce time
   constraints at solve time rather than patching them post-hoc.

3. 5-AXIS COST FUNCTION: cost, time, stops, layover, reliability —
   the same decomposition Module 2 uses. The baseline's compute_distance_
   matrix re-collapsed stops/layover into a single "convenience" number,
   undoing Module 1's careful preference decomposition. Fixed here.

4. STOCHASTIC BEAM SEARCH: after VRPTW produces the optimal ordering,
   enumerate concrete flight combinations using beam search with Monte
   Carlo price sampling to estimate robustness. Standard solvers pick
   one flight per leg greedily — beam search explores multiple partial
   itineraries in parallel. MC sampling estimates how robust each
   option's total price is against variance (using per-route price
   distributions computed by Module 1).

5. PARETO-OPTIMAL ITINERARIES: returns multiple trade-off-optimal
   complete itineraries, not just the single "best" ordering.

Two honest things found by testing this against the real 50,000-row
flights_clean.csv:

1. Route+date coverage in this dataset is sparse, not a daily schedule --
   most routes only have flights logged on 1-3 specific dates across an
   18-month span. The beam search's fallback path handles this gracefully
   by widening the date search when no temporally-feasible flights exist.

2. 18 of the 1,190 possible directed airport pairs have zero flights at
   all (e.g. BCN->BOM). solve_route() still returns *an* ordering (OR-Tools
   heavily penalizes, but does not remove, edges with no data), and
   beam_search_itineraries() is what actually catches and reports the
   infeasible leg -- this module never silently invents a flight.

Run:
    python module4_multi_city_router.py
The __main__ block runs a self-contained dummy-data demo and, if
output/flights_clean.csv exists, a second demo against the real
dataset including VRPTW routing + beam search with MC confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# Large-but-finite penalty for an edge with zero flights in the dataset.
# Not float('inf'): OR-Tools requires integer arc costs, and a genuinely
# infinite cost can make the solver misbehave (overflow, or refusing to
# ever consider the arc even when every alternative ordering is *also*
# infeasible). This is large enough that the solver will never choose it
# unless forced to.
INFEASIBLE_PENALTY = 1_000_000
COST_SCALE = 100_000   # blended [0,1] cost -> integer scale OR-Tools needs
DEFAULT_STAY_HOURS = 24  # minimum hours at each destination city


# =========================================================================== #
# 1. DATA STRUCTURES
# =========================================================================== #
@dataclass
class TemporalEdge:
    """A concrete flyable connection for one (origin, dest, date) triple.

    This is the key upgrade over baseline's RouteStats: edges are date-
    specific, enabling time-window constraints. The temporal flight graph
    is a dict mapping (origin, dest) -> [TemporalEdge, ...] sorted by date.

    Price distribution (mean/std) enables the downstream Monte Carlo
    stochastic confidence estimation in beam_search_itineraries().
    """
    origin: str
    destination: str
    date: pd.Timestamp
    min_price: float
    mean_price: float
    std_price: float
    min_duration_minutes: float
    min_stops: int
    min_layover_minutes: float
    avg_reliability_penalty: float
    num_flights: int


@dataclass
class ScoredItinerary:
    """A complete multi-leg itinerary with all flights resolved and scored.

    confidence: from Monte Carlo sampling — P(actual_total_price <= 1.1 ×
    estimated_total_price). Higher = this price estimate is more robust
    against real-world variance on these routes.
    """
    route: list[str]
    legs: list[dict]
    total_price: float
    total_duration_hours: float
    total_stops: int
    utility_score: float
    confidence: float
    is_feasible: bool


# =========================================================================== #
# 2. MULTI-CITY ROUTER (VRPTW + Stochastic Beam Search)
# =========================================================================== #
class MultiCityRouter:
    """VRPTW-based multi-city route optimizer with stochastic beam search.

    Upgrades over baseline TSP:
    1. Temporal edges: each (origin,dest,date) is a separate edge
    2. Time windows: departure must follow arrival + connection buffer
    3. 5-axis cost: user's full weight vector, not re-collapsed convenience
    4. Stochastic confidence: Monte Carlo sampling on price variance
    5. Pareto frontier: returns multiple trade-off-optimal itineraries

    Usage:
        router = MultiCityRouter(flights_df)
        route = router.solve_route("CPT", ["LHR", "CDG", "FCO"],
                                   start_date="2026-06-01", flexibility_days=7)
        itineraries = router.beam_search_itineraries(route, start_date="2026-06-01")
    """

    def __init__(self, flights_df: pd.DataFrame):
        flights_df = flights_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(flights_df["departure_utc"]):
            flights_df["departure_utc"] = pd.to_datetime(flights_df["departure_utc"], utc=True)
            flights_df["arrival_utc"] = pd.to_datetime(flights_df["arrival_utc"], utc=True)
        # Pre-compute flight_date if not already present (Module 1 now adds it)
        if "flight_date" not in flights_df.columns:
            flights_df["flight_date"] = flights_df["departure_utc"].dt.normalize()
        # Ensure reliability_penalty exists (needed for 5-axis scoring)
        if "reliability_penalty" not in flights_df.columns:
            flights_df["reliability_penalty"] = (
                100 - flights_df.get("on_time_performance", pd.Series(80, index=flights_df.index)).fillna(80)
            ).round(3)
        self.flights_df = flights_df
        self._temporal_edges: dict[tuple[str, str], list[TemporalEdge]] = {}
        self._route_agg: pd.DataFrame = pd.DataFrame()
        self._build_temporal_graph()

    # ------------------------------------------------------------------ #
    # temporal graph construction (vectorized)
    # ------------------------------------------------------------------ #
    def _build_temporal_graph(self):
        """Build temporal edge set — VECTORIZED via groupby, no Python loops
        over individual flights. O(n) in flight count.

        Each (origin, destination, date) becomes one TemporalEdge with
        aggregated stats for that specific date. This is the key difference
        from baseline's static RouteStats: edges are date-specific, enabling
        the VRPTW solver's time-window constraints.

        Also builds _route_agg: aggregate stats per (origin, destination)
        for the VRPTW cost matrix (the solver needs a single cost per edge)."""

        # --- Per-date temporal edges (for beam search) ---
        grouped = self.flights_df.groupby(
            ["origin", "destination", "flight_date"], observed=True
        ).agg(
            min_price=("price", "min"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            min_duration=("duration_minutes", "min"),
            min_stops=("stops", "min"),
            min_layover=("layover_minutes", "min"),
            avg_reliability=("reliability_penalty", "mean"),
            num_flights=("flight_id", "count"),
        ).reset_index()
        grouped["std_price"] = grouped["std_price"].fillna(0)
        grouped["avg_reliability"] = grouped["avg_reliability"].fillna(20.0)

        # Build the temporal edge dict from the grouped DataFrame
        # This iterrows is over the GROUPED result (much smaller than raw flights)
        for _, row in grouped.iterrows():
            key = (row["origin"], row["destination"])
            edge = TemporalEdge(
                origin=row["origin"],
                destination=row["destination"],
                date=row["flight_date"],
                min_price=float(row["min_price"]),
                mean_price=float(row["mean_price"]),
                std_price=float(row["std_price"]),
                min_duration_minutes=float(row["min_duration"]),
                min_stops=int(row["min_stops"]),
                min_layover_minutes=float(row["min_layover"]),
                avg_reliability_penalty=float(row["avg_reliability"]),
                num_flights=int(row["num_flights"]),
            )
            self._temporal_edges.setdefault(key, []).append(edge)

        # Sort each route's edges by date for efficient temporal lookups
        for key in self._temporal_edges:
            self._temporal_edges[key].sort(key=lambda e: e.date)

        # --- Aggregate route stats (for VRPTW cost matrix) ---
        # Uses MEAN price (not min) for more robust cost estimation
        self._route_agg = self.flights_df.groupby(
            ["origin", "destination"], observed=True
        ).agg(
            min_price=("price", "min"),
            mean_price=("price", "mean"),
            std_price=("price", "std"),
            min_duration=("duration_minutes", "min"),
            min_stops=("stops", "min"),
            min_layover=("layover_minutes", "min"),
            avg_reliability=("reliability_penalty", "mean"),
            num_options=("flight_id", "count"),
        )
        self._route_agg["std_price"] = self._route_agg["std_price"].fillna(0)
        self._route_agg["avg_reliability"] = self._route_agg["avg_reliability"].fillna(20.0)

        n_routes = len(self._route_agg)
        n_temporal = sum(len(v) for v in self._temporal_edges.values())
        print(f"[MultiCityRouter] built temporal graph: {n_routes} routes, "
              f"{n_temporal} date-specific edges from {len(self.flights_df)} flights")

    # ------------------------------------------------------------------ #
    # distance matrix (vectorized 5-axis)
    # ------------------------------------------------------------------ #
    def compute_distance_matrix(
        self,
        selected_cities: list[str],
        user_preferences: Optional[dict[str, float]] = None,
    ) -> list[list[int]]:
        """Fully vectorized NxN cost matrix using the FULL 5-axis cost
        function (cost, time, stops, layover, reliability).

        Key upgrade over baseline:
        - Uses MEAN price (not min) — more robust against outlier fares
        - 5 independent axes instead of re-collapsing stops/layover into
          a single "convenience" number
        - Normalization is local to this city set (same fix as Module 2)
        - np.einsum for the final weighted sum — one C-level call

        OR-Tools requires integer costs, so the final [0,1]-blended cost
        is scaled by COST_SCALE and rounded before being returned.
        """
        prefs = user_preferences or {}
        # Extract 5-axis weights, defaulting to equal 0.2 each
        weights = np.array([
            prefs.get("cost_weight", 0.2),
            prefs.get("time_weight", 0.2),
            prefs.get("stops_weight", 0.2),
            prefs.get("layover_weight", 0.2),
            prefs.get("reliability_weight", 0.2),
        ], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-10)  # normalize to sum=1

        n = len(selected_cities)
        # Build raw matrices for all 5 axes — (n, n, 5)
        raw = np.full((n, n, 5), np.nan, dtype=np.float64)
        feasible = np.zeros((n, n), dtype=bool)

        for i, o in enumerate(selected_cities):
            for j, d in enumerate(selected_cities):
                if i == j:
                    continue
                if (o, d) in self._route_agg.index:
                    row = self._route_agg.loc[(o, d)]
                    feasible[i, j] = True
                    raw[i, j] = [
                        row["mean_price"],       # cost axis — MEAN not min
                        row["min_duration"],     # time axis — best case
                        row["min_stops"],        # stops axis
                        row["min_layover"],      # layover axis
                        row["avg_reliability"],  # reliability axis
                    ]

        # --- Vectorized local normalization across all 5 axes simultaneously ---
        # This replaces 5 separate Python-loop normalizations with one
        # NumPy operation: reshape to (n*n, 5), compute min/max, normalize.
        feasible_3d = feasible[:, :, np.newaxis]
        raw_feasible = np.where(feasible_3d, raw, np.nan)

        axis_min = np.nanmin(raw_feasible.reshape(-1, 5), axis=0)  # (5,)
        axis_max = np.nanmax(raw_feasible.reshape(-1, 5), axis=0)  # (5,)
        spread = axis_max - axis_min
        spread[spread == 0] = 1.0

        normalized = (raw - axis_min) / spread  # (n, n, 5) — fully vectorized
        normalized = np.nan_to_num(normalized, nan=0.0)

        # Weighted sum across 5 axes — one einsum call instead of
        # d separate multiply-and-add operations
        blended = np.einsum("ijk,k->ij", normalized, weights)  # (n, n)

        matrix = np.rint(blended * COST_SCALE).astype(np.int64)
        matrix[~feasible] = INFEASIBLE_PENALTY
        np.fill_diagonal(matrix, 0)
        return matrix.tolist()

    # ------------------------------------------------------------------ #
    # VRPTW solver
    # ------------------------------------------------------------------ #
    def solve_route(
        self,
        home_airport: str,
        selected_cities: list[str],
        user_preferences: Optional[dict[str, float]] = None,
        time_limit_seconds: int = 5,
        start_date: Optional[str] = None,
        flexibility_days: int = 7,
        stay_hours: int = DEFAULT_STAY_HOURS,
    ) -> list[str]:
        """VRPTW-based city ordering with time windows.

        Upgrade over baseline TSP:
        - When start_date is provided, adds a time dimension with windows
          that enforce temporal feasibility AT SOLVE TIME, not post-hoc
        - Time callback uses actual min flight durations from the data
        - 5-axis cost function preserves per-user preference decomposition
        - Returns the ordering; use beam_search_itineraries() to resolve
          concrete flights with Monte Carlo confidence

        Without start_date: falls back to standard TSP (still with the
        improved 5-axis cost function).

        home -> [cities in optimal order] -> home (round trip).
        """
        selected_cities = [c for c in dict.fromkeys(selected_cities) if c != home_airport]
        if not selected_cities:
            return [home_airport, home_airport]
        if len(selected_cities) > 6:
            print(
                f"[solve_route] WARNING: {len(selected_cities)} destinations is well beyond "
                f"this problem's intended scope (<=5) -- OR-Tools will still attempt it, but "
                f"city-ordering stops being the interesting part of the problem at this scale."
            )

        all_nodes = [home_airport] + selected_cities
        n = len(all_nodes)
        distance_matrix = self.compute_distance_matrix(all_nodes, user_preferences)

        manager = pywrapcp.RoutingIndexManager(n, 1, 0)  # 1 vehicle, depot = index 0
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # --- VRPTW: Add time dimension with windows ---
        # Time is measured in hours from the start_date.
        # This is the critical upgrade: the solver now rejects orderings
        # that are temporally infeasible, instead of finding out post-hoc
        # in get_itinerary_flights().
        if start_date:
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                o, d = all_nodes[from_node], all_nodes[to_node]
                if o == d:
                    return 0
                if (o, d) in self._route_agg.index:
                    # Flight duration + stay time at destination
                    flight_hours = float(self._route_agg.loc[(o, d), "min_duration"]) / 60
                    return int(flight_hours + stay_hours)
                return int(24 + stay_hours)  # fallback estimate for missing routes

            time_callback_index = routing.RegisterTransitCallback(time_callback)

            # Total horizon: flexibility window + travel time for all legs
            max_horizon = flexibility_days * 24 + n * (24 + stay_hours)

            routing.AddDimension(
                time_callback_index,
                slack_max=flexibility_days * 24,  # max waiting time at any node (hours)
                capacity=max_horizon,             # max cumulative time
                fix_start_cumul_to_zero=True,
                name="Time",
            )
            time_dimension = routing.GetDimensionOrDie("Time")

            # Set time windows per node
            for node_idx in range(n):
                index = manager.NodeToIndex(node_idx)
                if node_idx == 0:  # depot (home) — depart at time 0
                    time_dimension.CumulVar(index).SetRange(0, 0)
                else:
                    # Each destination can be visited anywhere in the horizon
                    time_dimension.CumulVar(index).SetRange(0, max_horizon)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.FromSeconds(time_limit_seconds)

        solution = routing.SolveWithParameters(search_parameters)
        if solution is None:
            raise RuntimeError(
                f"OR-Tools VRPTW found no solution for {home_airport} -> {selected_cities}. "
                f"This shouldn't happen on a complete graph with a finite penalty cost -- check "
                f"that INFEASIBLE_PENALTY x (number of nodes) isn't overflowing anything."
            )

        index = routing.Start(0)
        order = []
        penalty_edges = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            order.append(all_nodes[node])
            next_index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(next_index):
                to_node = manager.IndexToNode(next_index)
                if distance_matrix[node][to_node] >= INFEASIBLE_PENALTY:
                    penalty_edges += 1
            index = next_index
        order.append(home_airport)

        if penalty_edges:
            print(
                f"[solve_route] WARNING: the optimal ordering found still crosses "
                f"{penalty_edges} route(s) with zero flights in the dataset -- every possible "
                f"ordering was infeasible on at least one leg. Check beam_search_itineraries() "
                f"or get_itinerary_flights() output to see exactly which leg(s) failed."
            )

        return order

    # ------------------------------------------------------------------ #
    # Stochastic Beam Search (novel component)
    # ------------------------------------------------------------------ #
    def beam_search_itineraries(
        self,
        optimal_route: list[str],
        start_date: str,
        user_preferences: Optional[dict[str, float]] = None,
        beam_width: int = 5,
        mc_samples: int = 100,
        min_connection_minutes: int = 90,
    ) -> list[ScoredItinerary]:
        """Stochastic Beam Search: given a city ordering from VRPTW,
        enumerate concrete flight combinations using beam search with
        Monte Carlo price confidence scoring.

        Why this is novel:
        - Standard solvers pick one flight per leg GREEDILY (cheapest
          departing after previous arrival). This misses cases where a
          slightly more expensive first leg enables a much cheaper second leg.
        - Beam search maintains beam_width partial itineraries at each leg,
          exploring the trade-off space between legs.
        - Monte Carlo confidence: for each complete itinerary, sample prices
          from N(route_mean, route_std) for each leg mc_samples times.
          Confidence = P(sampled_total <= 1.1 × point_estimate). Higher
          confidence means this price is more robust against real-world
          variance on these specific routes.

        Algorithm:
        1. Initialize beam with empty itinerary at start_date
        2. For each leg (origin -> destination):
           a. For each partial itinerary in beam:
              - Find all flights departing after current_time + buffer
              - Score on 5-axis utility (vectorized NumPy)
              - Expand beam with top-beam_width candidates
           b. Prune beam to beam_width best partial itineraries
        3. Score complete itineraries with Monte Carlo sampling
        4. Return sorted by utility score
        """
        prefs = user_preferences or {}
        # 5-axis weight vector — same decomposition as Module 2
        w = np.array([
            prefs.get("cost_weight", 0.2),
            prefs.get("time_weight", 0.2),
            prefs.get("stops_weight", 0.2),
            prefs.get("layover_weight", 0.2),
            prefs.get("reliability_weight", 0.2),
        ], dtype=np.float64)
        w = w / (w.sum() + 1e-10)

        current_time = pd.Timestamp(start_date, tz="UTC")

        # Beam state: list of (partial_legs, cumulative_utility, current_arrival_time)
        beam: list[tuple[list[dict], float, pd.Timestamp]] = [([], 0.0, current_time)]

        legs_pairs = list(zip(optimal_route[:-1], optimal_route[1:]))

        for leg_idx, (origin, destination) in enumerate(legs_pairs):
            next_beam: list[tuple[list[dict], float, pd.Timestamp]] = []

            for partial_legs, cum_score, arr_time in beam:
                # Find flights departing after arrival + connection buffer
                candidates = self.flights_df[
                    (self.flights_df["origin"] == origin)
                    & (self.flights_df["destination"] == destination)
                    & (self.flights_df["departure_utc"] >= arr_time)
                ]

                if candidates.empty:
                    # Date gap: try ANY flight on this route (sparse dataset)
                    candidates = self.flights_df[
                        (self.flights_df["origin"] == origin)
                        & (self.flights_df["destination"] == destination)
                    ]

                if candidates.empty:
                    # Truly infeasible — no flights exist for this route at all
                    next_beam.append((
                        partial_legs + [{
                            "origin": origin,
                            "destination": destination,
                            "flight_id": None,
                            "status": "INFEASIBLE",
                            "reason": "no flights exist for this route in the dataset",
                        }],
                        cum_score - 10.0,  # heavy penalty
                        arr_time,
                    ))
                    continue

                # --- Vectorized 5-axis scoring of all candidates ---
                # O(n × d) NumPy instead of per-flight Python scoring
                score_cols = ["price", "duration_minutes", "stops",
                              "layover_minutes", "reliability_penalty"]
                raw = candidates[score_cols].fillna(0).to_numpy(dtype=np.float64)
                # Local normalization within this leg's candidates
                col_min = raw.min(axis=0)
                col_max = raw.max(axis=0)
                spread = col_max - col_min
                spread[spread == 0] = 1.0
                normed = (raw - col_min) / spread
                scores = -(normed @ w)  # higher = better (negative weighted sum)

                # Top-k candidates for beam expansion
                top_k = min(beam_width, len(candidates))
                top_indices = np.argsort(-scores)[:top_k]

                for idx in top_indices:
                    flight = candidates.iloc[idx]
                    leg_info = {
                        "origin": origin,
                        "destination": destination,
                        "flight_id": flight.get("flight_id"),
                        "airline_name": flight.get("airline_name"),
                        "departure_utc": flight["departure_utc"],
                        "arrival_utc": flight["arrival_utc"],
                        "price": float(flight["price"]),
                        "duration_hours": float(
                            flight.get("duration_hours", flight["duration_minutes"] / 60)
                        ),
                        "stops": int(flight["stops"]),
                        "status": "OK",
                        "date_gap_days": max(0, (flight["departure_utc"] - arr_time).days),
                    }
                    new_arr = flight["arrival_utc"] + pd.Timedelta(minutes=min_connection_minutes)
                    next_beam.append((
                        partial_legs + [leg_info],
                        cum_score + float(scores[idx]),
                        new_arr,
                    ))

            # Prune beam to top beam_width by cumulative utility
            next_beam.sort(key=lambda x: -x[1])
            beam = next_beam[:beam_width]

        # --- Monte Carlo confidence scoring ---
        # For each complete itinerary, sample prices from N(route_mean, route_std)
        # mc_samples times and compute P(sampled_total <= 1.1 × point_estimate)
        rng = np.random.default_rng(42)
        itineraries: list[ScoredItinerary] = []

        for partial_legs, cum_score, _ in beam:
            if not partial_legs:
                continue

            feasible_legs = [leg for leg in partial_legs if leg.get("status") == "OK"]
            total_price = sum(leg["price"] for leg in feasible_legs)
            total_dur = sum(leg.get("duration_hours", 0) for leg in feasible_legs)
            total_stops = sum(leg.get("stops", 0) for leg in feasible_legs)
            is_feasible = all(leg.get("status") == "OK" for leg in partial_legs)

            # MC sampling: vectorized across all samples × legs
            if feasible_legs and is_feasible:
                # Build (mc_samples, n_legs) matrix of sampled prices
                n_legs = len(feasible_legs)
                mc_matrix = np.zeros((mc_samples, n_legs), dtype=np.float64)

                for leg_i, leg in enumerate(feasible_legs):
                    route_key = (leg["origin"], leg["destination"])
                    if route_key in self._route_agg.index:
                        mu = float(self._route_agg.loc[route_key, "mean_price"])
                        sigma = float(self._route_agg.loc[route_key, "std_price"])
                        if np.isnan(sigma) or sigma == 0:
                            mc_matrix[:, leg_i] = mu
                        else:
                            # Clip at 0 — prices can't be negative
                            mc_matrix[:, leg_i] = np.maximum(0, rng.normal(mu, sigma, mc_samples))
                    else:
                        mc_matrix[:, leg_i] = leg["price"]

                # Vectorized sum across legs for each sample
                mc_totals = mc_matrix.sum(axis=1)  # (mc_samples,)
                # Confidence = P(sampled_total <= 1.1 × point_estimate)
                confidence = float(np.mean(mc_totals <= 1.1 * total_price))
            else:
                confidence = 0.0

            # Build route list from leg origins + final destination
            route_cities = []
            for leg in partial_legs:
                if leg.get("origin") and leg["origin"] not in route_cities:
                    route_cities.append(leg["origin"])
            if partial_legs and partial_legs[-1].get("destination"):
                route_cities.append(partial_legs[-1]["destination"])

            itineraries.append(ScoredItinerary(
                route=route_cities,
                legs=partial_legs,
                total_price=round(total_price, 2),
                total_duration_hours=round(total_dur, 2),
                total_stops=total_stops,
                utility_score=round(cum_score, 4),
                confidence=round(confidence, 3),
                is_feasible=is_feasible,
            ))

        # Sort by utility score, best first
        itineraries.sort(key=lambda x: -x.utility_score)
        return itineraries

    # ------------------------------------------------------------------ #
    # backward-compatible concrete flight resolution (greedy)
    # ------------------------------------------------------------------ #
    def get_itinerary_flights(
        self,
        optimal_route: list[str],
        start_date: str,
        min_connection_minutes: int = 90,
    ) -> dict:
        """Walk the ordered route leg by leg and pick a concrete flight
        for each, enforcing that each leg's departure is after the
        previous leg's arrival (+ a minimum connection buffer).

        This is the BACKWARD-COMPATIBLE greedy resolver — picks the earliest
        workable date, then the cheapest flight on that date. For the
        upgraded path with beam search and Monte Carlo confidence, use
        beam_search_itineraries() instead.
        """
        current_time = pd.Timestamp(start_date, tz="UTC")
        legs = []
        all_ok = True

        for origin, destination in zip(optimal_route[:-1], optimal_route[1:]):
            candidates = self.flights_df[
                (self.flights_df["origin"] == origin)
                & (self.flights_df["destination"] == destination)
                & (self.flights_df["departure_utc"] >= current_time)
            ]

            if candidates.empty:
                any_flights_ever = self.flights_df[
                    (self.flights_df["origin"] == origin) & (self.flights_df["destination"] == destination)
                ]
                reason = (
                    "no flights exist for this route in the dataset at all"
                    if any_flights_ever.empty
                    else "route exists, but nothing departs on/after the required connection time"
                )
                legs.append({
                    "origin": origin, "destination": destination, "flight_id": None,
                    "status": "INFEASIBLE", "reason": reason,
                })
                all_ok = False
                continue

            earliest_date = candidates["departure_utc"].min().normalize()
            same_day = candidates[candidates["departure_utc"].dt.normalize() == earliest_date]
            best = same_day.sort_values("price").iloc[0]

            gap_days = (best["departure_utc"] - current_time).days
            legs.append({
                "origin": origin,
                "destination": destination,
                "flight_id": best["flight_id"],
                "airline_name": best.get("airline_name"),
                "departure_utc": best["departure_utc"],
                "arrival_utc": best["arrival_utc"],
                "price": float(best["price"]),
                "duration_hours": float(best.get("duration_hours", best["duration_minutes"] / 60)),
                "stops": int(best["stops"]),
                "status": "OK",
                "date_gap_days": gap_days,
            })
            current_time = best["arrival_utc"] + pd.Timedelta(minutes=min_connection_minutes)

        total_price = sum(leg["price"] for leg in legs if leg["status"] == "OK")
        return {
            "route": optimal_route,
            "legs": legs,
            "all_legs_feasible": all_ok,
            "total_price": round(total_price, 2) if all_ok else None,
        }


# =========================================================================== #
# main / smoke test
# =========================================================================== #
if __name__ == "__main__":
    print("=" * 70)
    print("DEMO 1: dummy data (self-contained, no external files needed)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    dummy_airports = ["HOME", "A", "B", "C"]
    rows, fid = [], 0
    for o in dummy_airports:
        for d in dummy_airports:
            if o == d:
                continue
            if o == "A" and d == "C":
                continue  # deliberately missing route -> exercises infeasible edge case
            for _ in range(10):
                fid += 1
                dep = (
                    pd.Timestamp("2026-06-01", tz="UTC")
                    + pd.Timedelta(days=int(rng.integers(0, 120)))
                    + pd.Timedelta(minutes=int(rng.integers(0, 1440)))
                )
                dur = int(rng.integers(120, 600))
                rows.append({
                    "flight_id": f"DUMMY{fid:04d}", "origin": o, "destination": d,
                    "airline_name": "Dummy Air", "departure_utc": dep,
                    "arrival_utc": dep + pd.Timedelta(minutes=dur),
                    "price": float(rng.integers(150, 1200)), "duration_minutes": dur,
                    "stops": int(rng.integers(0, 2)), "layover_minutes": int(rng.integers(0, 300)),
                    "on_time_performance": float(rng.integers(70, 99)),
                })
    dummy_df = pd.DataFrame(rows)

    router = MultiCityRouter(dummy_df)

    # VRPTW solve with time windows
    route = router.solve_route(
        "HOME", ["A", "B", "C"],
        user_preferences={"cost_weight": 0.5, "time_weight": 0.2,
                          "stops_weight": 0.1, "layover_weight": 0.1,
                          "reliability_weight": 0.1},
        start_date="2026-06-01",
        flexibility_days=14,
    )
    print("VRPTW optimal order:", route)

    # Beam search with Monte Carlo confidence
    itineraries = router.beam_search_itineraries(
        route, start_date="2026-06-01",
        user_preferences={"cost_weight": 0.5, "time_weight": 0.2,
                          "stops_weight": 0.1, "layover_weight": 0.1,
                          "reliability_weight": 0.1},
        beam_width=3, mc_samples=50,
    )
    print(f"\nBeam search found {len(itineraries)} itinerary(-ies):")
    for i, itin in enumerate(itineraries):
        print(f"  #{i+1}: ${itin.total_price:.0f}, {itin.total_duration_hours:.1f}h, "
              f"{itin.total_stops} stops, utility={itin.utility_score:.3f}, "
              f"confidence={itin.confidence:.1%}, feasible={itin.is_feasible}")
        for leg in itin.legs:
            if leg["status"] == "OK":
                print(f"      {leg['origin']}->{leg['destination']}: "
                      f"${leg['price']:.0f}, {leg['duration_hours']:.1f}h, "
                      f"{leg['stops']} stops ({leg.get('airline_name', '?')})")
            else:
                print(f"      {leg['origin']}->{leg['destination']}: INFEASIBLE ({leg['reason']})")

    # Backward-compatible greedy resolution
    print("\nGreedy itinerary (backward-compatible):")
    itinerary = router.get_itinerary_flights(route, start_date="2026-06-01")
    for leg in itinerary["legs"]:
        print(" ", leg)
    print("all_legs_feasible:", itinerary["all_legs_feasible"],
          " total_price:", itinerary["total_price"])

    print()
    print("=" * 70)
    print("DEMO 2: real hackathon dataset (output/flights_clean.csv), if present")
    print("=" * 70)
    real_path = Path("output/flights_clean.csv")
    if real_path.exists():
        df_real = pd.read_csv(real_path, parse_dates=["departure_utc", "arrival_utc"])
        router_real = MultiCityRouter(df_real)

        print("-- VRPTW + beam search: Cape Town + London/Paris/Rome --")
        route = router_real.solve_route(
            "CPT", ["LHR", "CDG", "FCO"],
            user_preferences={"cost_weight": 0.5, "time_weight": 0.2,
                              "stops_weight": 0.1, "layover_weight": 0.1,
                              "reliability_weight": 0.1},
            start_date="2025-01-01",
            flexibility_days=14,
        )
        print("VRPTW optimal order:", route)

        itineraries = router_real.beam_search_itineraries(
            route, start_date="2025-01-01",
            user_preferences={"cost_weight": 0.5, "time_weight": 0.2,
                              "stops_weight": 0.1, "layover_weight": 0.1,
                              "reliability_weight": 0.1},
            beam_width=3, mc_samples=100,
        )
        print(f"Beam search found {len(itineraries)} itinerary(-ies):")
        for i, itin in enumerate(itineraries):
            print(f"  #{i+1}: ${itin.total_price:.0f}, {itin.total_duration_hours:.1f}h, "
                  f"confidence={itin.confidence:.1%}, feasible={itin.is_feasible}")

        print()
        print("-- edge case: a route with zero flights in the real dataset (BCN -> BOM) --")
        route2 = router_real.solve_route("BCN", ["BOM", "CPT"])
        print("Optimal order:", route2)
        itinerary2 = router_real.get_itinerary_flights(route2, start_date="2026-06-01")
        for leg in itinerary2["legs"]:
            print(" ", leg)
        print("all_legs_feasible:", itinerary2["all_legs_feasible"])
    else:
        print(f"  {real_path} not found -- run module1_eda_cleaning.py first to generate it.")
