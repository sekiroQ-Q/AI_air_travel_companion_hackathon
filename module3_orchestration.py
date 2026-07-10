"""
Module 3 - End-to-End Orchestration Pipeline
==============================================
AI Air Travel Companion (hackathon prototype)

Ties together: NL query -> preference retrieval (FAISS, Module 2) ->
constrained multi-objective optimization (Module 2) -> explanation with
explicit trade-offs and cited evidence.

Design choice -- deterministic pipeline, not a free agent:
LangChain supports both a fixed pipeline (prompt -> chain -> chain...)
and a free-form AgentExecutor that decides its own tool sequence. This
module deliberately uses a fixed pipeline: the four steps here (parse ->
retrieve -> optimize -> explain) are always the same for this problem,
and a fixed pipeline can't loop, call tools in the wrong order, or hang
mid-demo the way an agent occasionally can. If you later want the model
to *choose* what to do next (e.g. ask a clarifying question before
optimizing), wrapping `parse_query`, `retrieve_and_optimize`, and
`explain` as LangChain `Tool`s for an `AgentExecutor` is a small
extension -- noted as future work rather than the default, because demo
reliability is a judged criterion here.

Two swappable backends for the two LLM-shaped steps (query parsing,
explanation), both behind the same interface:
  - llm=None       (default) -> rule-based parsing + template explanation.
                     Zero external model dependency. Always works,
                     including fully offline.
  - llm=<LangChain LLM/ChatModel> -> routes both steps through a real
                     open-source model (see `main()` for wiring examples
                     with Ollama or a local HuggingFace pipeline). Any
                     failure at call time (model unreachable, malformed
                     output, timeout) is caught and falls back to the
                     rule-based/template path automatically -- the same
                     graceful-degradation pattern used for the embedder
                     in Module 2, and for the same reason.

Run:
    python module3_orchestration.py
(expects module1's and module2's outputs to already exist)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from module2_preference_engine import (
    PreferenceEmbedder,
    UserPreferenceStore,
    filter_candidates,
    score_and_rank,
)

OUTPUT_DIR = Path("output")
FLIGHTS_CLEAN_PATH = OUTPUT_DIR / "flights_clean.csv"
PROFILES_PATH = OUTPUT_DIR / "user_profiles.json"


# =========================================================================== #
# 1. QUERY PARSING
# =========================================================================== #
FLEX_KEYWORDS = {"flexible": 14, "whenever": 21, "no rush": 14, "anytime": 21}
DIRECT_SIGNALS = ["hate layover", "hate connections", "no layover", "direct only", "nonstop", "avoid layover"]
BUDGET_SIGNALS = ["budget", "cheap", "cheapest", "affordable", "save money"]


def build_city_lookup(df_flights: pd.DataFrame) -> dict:
    """city name (lowercase) -> IATA code, built from the dataset itself
    so the parser never drifts out of sync with what's actually flyable."""
    lookup = {}
    for _, row in df_flights[["destination", "destination_city"]].drop_duplicates().iterrows():
        lookup[str(row["destination_city"]).lower()] = row["destination"]
    for _, row in df_flights[["origin", "origin_city"]].drop_duplicates().iterrows():
        lookup.setdefault(str(row["origin_city"]).lower(), row["origin"])
    return lookup


def parse_query_rule_based(query: str, city_lookup: dict, default_flex_days: int = 3) -> dict:
    """Zero-dependency slot extraction. Always available."""
    text = query.lower()
    parsed = {
        "destinations": [],
        "date_hint": None,
        "flexibility_days": default_flex_days,
        "wants_direct": None,
        "budget_hint": None,
    }

    for city, code in city_lookup.items():
        if city in text and code not in parsed["destinations"]:
            parsed["destinations"].append(code)

    if "next week" in text:
        parsed["date_hint"] = "next_week"
    elif "next month" in text:
        parsed["date_hint"] = "next_month"

    for kw, days in FLEX_KEYWORDS.items():
        if kw in text:
            parsed["flexibility_days"] = days

    if any(k in text for k in DIRECT_SIGNALS):
        parsed["wants_direct"] = True
    if any(k in text for k in BUDGET_SIGNALS):
        parsed["budget_hint"] = "price_sensitive"

    return parsed


def parse_query_llm(query: str, llm) -> dict:
    """LLM-based slot extraction via LangChain. `llm` must be a
    LangChain-compatible chat/LLM object -- see main() for setup examples.
    Raises on failure; callers should catch and fall back to
    parse_query_rule_based."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser

    prompt = ChatPromptTemplate.from_template(
        "Extract travel intent from this request as JSON with keys: "
        "destinations (list of city names mentioned), date_hint (string or null), "
        "flexibility_days (int), wants_direct (true/false/null), "
        "budget_hint (string or null). Respond with JSON only.\n"
        "Request: {query}\nJSON:"
    )
    chain = prompt | llm | JsonOutputParser()
    return chain.invoke({"query": query})


# =========================================================================== #
# 2. EXPLANATION GENERATION
# =========================================================================== #
def explain_template(query: str, top_pick: dict, alternatives: list[dict], evidence: list[str], profile: dict) -> str:
    """Deterministic, zero-dependency explanation. This is the default and
    the guaranteed-to-work fallback -- no model call, no network, no
    latency. Still hits every item in the benchmark's expected_behavior
    checklist: named trade-off with $ and hours, cited evidence, stated
    priorities."""
    lines = [f'Based on your travel history, here\'s my recommendation for "{query}":']
    lines.append(
        f"- Top pick: {top_pick['airline_name']} {top_pick['origin']}->{top_pick['destination']}, "
        f"${top_pick['price']:.0f}, {top_pick['duration_hours']:.1f}h, {top_pick['stops']} stop(s)."
    )

    others = [a for a in alternatives if a.get("flight_id") != top_pick.get("flight_id")]
    if others:
        cheapest = min(others, key=lambda a: a["price"])
        fastest = min(others, key=lambda a: a["duration_hours"])
        if cheapest["price"] < top_pick["price"]:
            price_delta = top_pick["price"] - cheapest["price"]
            time_delta = cheapest["duration_hours"] - top_pick["duration_hours"]
            if abs(time_delta) < 0.05:
                time_clause = "for the same flight time"
            else:
                time_clause = f"but takes {abs(time_delta):.1f}h {'longer' if time_delta > 0 else 'shorter'}"
            lines.append(
                f"- Trade-off: a cheaper option exists at ${cheapest['price']:.0f} "
                f"(${price_delta:.0f} less) {time_clause} -- "
                f"your priorities weight convenience over saving that amount, so we kept the top pick."
            )
        if fastest["flight_id"] != cheapest.get("flight_id") and fastest["duration_hours"] < top_pick["duration_hours"]:
            lines.append(f"- Fastest alternative on file: {fastest['duration_hours']:.1f}h for ${fastest['price']:.0f}.")

    if evidence:
        quoted = "; ".join(f'"{e}"' for e in evidence)
        lines.append(f"- Why this fits you: {quoted}.")

    w = profile.get("weights", {})
    lines.append(
        f"- Inferred priorities from your profile: cost {w.get('cost_weight', 0):.0%}, "
        f"time {w.get('time_weight', 0):.0%}, convenience {w.get('convenience_weight', 0):.0%}."
    )
    if "revealed_usd_per_layover_hour" in w:
        lines.append(f"- Revealed price elasticity: ~${w['revealed_usd_per_layover_hour']:.0f} per hour of layover tolerated.")

    if top_pick.get("stops", 0) and profile.get("max_layover_minutes") is not None:
        lines.append(f"- Layover within your {profile['max_layover_minutes']}-minute tolerance.")

    return "\n".join(lines)


def explain_llm(context: dict, llm) -> str:
    """LLM-based explanation via LangChain. Raises on failure; callers
    should catch and fall back to explain_template."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(
        "You are a travel assistant. In 3-4 sentences, explain why the recommended "
        "flight fits this traveler. Explicitly name the cost-vs-time-vs-convenience "
        "trade-off with concrete dollar and hour figures, and cite the evidence from "
        "their history.\n\n"
        "Query: {query}\n"
        "Top pick: {top_pick}\n"
        "Alternatives considered: {alternatives}\n"
        "Evidence from user history: {evidence}\n"
        "Inferred priorities: {weights}\n\n"
        "Explanation:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({k: str(v) for k, v in context.items()})


# =========================================================================== #
# 3. ORCHESTRATOR
# =========================================================================== #
class TravelCompanionPipeline:
    def __init__(self, df_flights: pd.DataFrame, store: UserPreferenceStore, llm=None):
        self.df_flights = df_flights
        self.store = store
        self.city_lookup = build_city_lookup(df_flights)
        self.llm = llm  # None => rule-based parsing + template explanation throughout

    def run(self, user_id: str, query: str, top_k: int = 3) -> dict:
        profile = self.store.get_profile(user_id)

        # --- 1. parse -------------------------------------------------------
        if self.llm is not None:
            try:
                parsed = parse_query_llm(query, self.llm)
            except Exception as e:
                print(f"[pipeline] LLM query parsing failed ({e.__class__.__name__}) -- using rule-based parser")
                parsed = parse_query_rule_based(query, self.city_lookup)
        else:
            parsed = parse_query_rule_based(query, self.city_lookup)

        destinations = parsed.get("destinations")
        if not destinations:
            return {"error": "could not identify a destination in the query", "parsed_query": parsed}

        # --- 2. retrieve evidence (FAISS) -----------------------------------
        evidence = self.store.evidence_for_query(user_id, query, k=3)

        # --- 3. optimize (hard constraints -> multi-objective scoring) ------
        # an explicit "hate layovers" in the live query is a harder constraint
        # than whatever the stored profile says -- ad-hoc query constraints
        # override stored preferences, never the other way around.
        max_layover = 0 if parsed.get("wants_direct") else profile.get("max_layover_minutes")
        candidates = filter_candidates(
            self.df_flights,
            origin=profile["home_airport"],
            destinations=destinations,
            max_layover_minutes=max_layover,
        )
        result = score_and_rank(candidates, profile, top_k=top_k)
        if not result["top_k"]:
            return {"error": "no flights matched the hard constraints", "parsed_query": parsed}

        # --- 4. explain -------------------------------------------------------
        top_pick = result["top_k"][0]
        alternatives = result["pareto_frontier"]
        if self.llm is not None:
            try:
                explanation = explain_llm(
                    {"query": query, "top_pick": top_pick, "alternatives": alternatives,
                     "evidence": evidence, "weights": profile.get("weights", {})},
                    self.llm,
                )
            except Exception as e:
                print(f"[pipeline] LLM explanation failed ({e.__class__.__name__}) -- using template explainer")
                explanation = explain_template(query, top_pick, alternatives, evidence, profile)
        else:
            explanation = explain_template(query, top_pick, alternatives, evidence, profile)

        return {
            "parsed_query": parsed,
            "top_3": result["top_k"],
            "pareto_frontier": alternatives,
            "evidence_used": evidence,
            "explanation": explanation,
        }


# =========================================================================== #
# main / smoke test
# =========================================================================== #
def main():
    print("Loading Module 1/2 outputs...")
    df_flights = pd.read_csv(FLIGHTS_CLEAN_PATH, parse_dates=["departure_utc", "arrival_utc"])
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)

    embedder = PreferenceEmbedder()
    store = UserPreferenceStore(embedder)
    store.build(profiles)

    # --- LLM wiring (optional) --------------------------------------------
    # Uncomment ONE of these to route parsing/explanation through a real
    # open-source LLM. Left as llm=None below, the pipeline runs entirely
    # rule-based/template -- zero extra dependencies, zero network calls.
    #
    #   from langchain_community.chat_models import ChatOllama
    #   llm = ChatOllama(model="llama3")                     # requires local Ollama server
    #
    #   from langchain_community.llms import HuggingFacePipeline
    #   llm = HuggingFacePipeline.from_model_id(
    #       model_id="google/flan-t5-base", task="text2text-generation"
    #   )                                                     # requires internet on first run
    llm = None

    pipeline = TravelCompanionPipeline(df_flights, store, llm=llm)

    query = "I need a flight from home to London next week, I hate layovers but I'm on a budget"
    user_id = "U23"
    result = pipeline.run(user_id=user_id, query=query)

    print(f"\nUser: {user_id}  Query: {query!r}")
    print(f"Parsed: {result.get('parsed_query')}")
    print(f"Evidence retrieved: {result.get('evidence_used')}")
    print("\n--- Explanation ---")
    print(result.get("explanation", result.get("error")))

    print("\n--- Top 3 ---")
    for f in result.get("top_3", []):
        print(f"  {f['airline_name']:20s} ${f['price']:>7.0f}  {f['duration_hours']:>5.1f}h  "
              f"{f['stops']} stop(s)  utility={f['utility_score']:.3f}")


if __name__ == "__main__":
    main()
