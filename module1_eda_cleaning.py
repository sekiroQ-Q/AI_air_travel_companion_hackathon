"""
Module 1 - EDA & Data Cleaning
==============================
AI Air Travel Companion (hackathon prototype)

Responsibilities:
  1. Clean df_flights: handle missing values, normalize price/duration,
     engineer the numeric features the Module 2 optimizer needs.
  2. Mine df_users' unstructured `raw_history` text with regex + light
     rule-based NLP to extract preferences that aren't already in the
     structured columns (budget stance, direct-flight stance, redeye
     tolerance, revealed price elasticity, airline mentions, etc).
  3. Fuse structured columns + mined signals into a per-user numeric
     multi-objective weight vector, and a natural-language "preference
     summary" string that Module 2 will embed.
  4. Save everything Module 2/3 need, plus a small EDA plot set.

Expected input layout (adjust DATA_DIR if yours differs):
    data/flights_data.csv
    data/user_data.csv

Outputs:
    output/flights_clean.csv       -- cleaned + feature-engineered flights
    output/user_profiles.json      -- one fused profile per user
    output/eda_plots/*.png         -- five diagnostic plots

Run:
    python module1_eda_cleaning.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never opens a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
PLOTS_DIR = OUTPUT_DIR / "eda_plots"

FLIGHTS_PATH = DATA_DIR / "flights_data.csv"
USERS_PATH = DATA_DIR / "user_data.csv"

sns.set_theme(style="whitegrid")


# =========================================================================== #
# 1. FLIGHT DATA CLEANING
# =========================================================================== #
def load_flights(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, fix dtypes, and engineer the features the
    Module 2 multi-objective optimizer scores flights on."""
    df = df.copy()

    # --- missing values --------------------------------------------------
    # layover_airports / layover_minutes are legitimately empty for direct
    # flights (stops == 0). That's not missing data -- make it explicit
    # rather than dropping or imputing it.
    df["layover_airports"] = df["layover_airports"].fillna("")
    df["layover_minutes"] = df["layover_minutes"].fillna(0)

    # fields the optimizer cannot function without -> drop if truly absent
    critical_cols = ["origin", "destination", "price", "duration_minutes", "stops"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    if before - len(df):
        print(f"[clean_flights] dropped {before - len(df)} rows missing critical fields")

    # --- dtype normalization ----------------------------------------------
    df["departure_utc"] = pd.to_datetime(df["departure_utc"], utc=True, errors="coerce")
    df["arrival_utc"] = pd.to_datetime(df["arrival_utc"], utc=True, errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df["stops"] = pd.to_numeric(df["stops"], errors="coerce").astype("Int64")
    df["seats_available"] = pd.to_numeric(df.get("seats_available"), errors="coerce")
    df["on_time_performance"] = pd.to_numeric(df.get("on_time_performance"), errors="coerce")

    for bool_col in ["baggage_included", "refundable", "is_holiday_season"]:
        if bool_col in df.columns:
            df[bool_col] = (
                df[bool_col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .fillna(False)
            )

    # currency: normalize defensively even though this dataset is USD-only,
    # so the pipeline doesn't silently break if a mixed-currency file is
    # swapped in later.
    if "currency" in df.columns:
        non_usd = df.loc[df["currency"] != "USD"]
        if len(non_usd):
            print(f"[clean_flights] WARNING: {len(non_usd)} rows in non-USD currency "
                  f"were left unconverted -- add an FX step if this matters for your data.")

    # --- feature engineering ------------------------------------------------
    df["duration_hours"] = (df["duration_minutes"] / 60).round(2)
    df["is_direct"] = df["stops"] == 0
    df["price_per_hour"] = (df["price"] / df["duration_hours"].replace(0, np.nan)).round(2)

    df["departure_hour_utc"] = df["departure_utc"].dt.hour
    df["is_redeye"] = df["departure_hour_utc"].between(0, 5, inclusive="both")

    df["layover_hours"] = (df["layover_minutes"] / 60).round(2)
    # a single "inconvenience" axis blending stop count, total layover
    # exposure, and on-time reliability -- one of the three optimizer axes
    df["inconvenience_score"] = (
        df["stops"].astype(float) * 2
        + df["layover_hours"] * 0.5
        + (100 - df["on_time_performance"].fillna(80)) * 0.05
    ).round(3)

    # min-max normalize the three optimizer axes to [0, 1] once here, so
    # Module 2 never has to re-derive scale-sensitive constants per query
    for col in ["price", "duration_minutes", "inconvenience_score"]:
        cmin, cmax = df[col].min(), df[col].max()
        df[f"{col}_norm"] = ((df[col] - cmin) / (cmax - cmin)).round(4) if cmax > cmin else 0.0

    before = len(df)
    df = df.drop_duplicates(
        subset=["origin", "destination", "departure_utc", "airline_code", "cabin_class", "price"]
    )
    if before - len(df):
        print(f"[clean_flights] dropped {before - len(df)} duplicate itineraries")

    return df.reset_index(drop=True)


def plot_flight_eda(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["price"], bins=50, ax=ax, color="#378ADD")
    ax.set_title("Flight price distribution")
    ax.set_xlabel("Price (USD)")
    fig.tight_layout()
    fig.savefig(plots_dir / "01_price_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    order = df.groupby("season")["price"].mean().sort_values(ascending=False).index
    sns.boxplot(data=df, x="season", y="price", order=order, ax=ax)
    ax.set_title("Price by season")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots_dir / "02_price_by_season.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="stops", y="price", ax=ax)
    ax.set_title("Price by number of stops")
    fig.tight_layout()
    fig.savefig(plots_dir / "03_price_by_stops.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    sample = df.sample(min(4000, len(df)), random_state=42)
    sns.scatterplot(
        data=sample, x="duration_hours", y="price", hue="is_direct",
        alpha=0.4, s=18, ax=ax, palette={True: "#0F6E56", False: "#D85A30"},
    )
    ax.set_title("Price vs. duration (direct vs. connecting)")
    fig.tight_layout()
    fig.savefig(plots_dir / "04_price_vs_duration.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    top_routes = (
        df.assign(route=df["origin"] + "-" + df["destination"])
        .groupby("route").size().sort_values(ascending=False).head(10)
    )
    sns.barplot(x=top_routes.values, y=top_routes.index, ax=ax, color="#7F77DD")
    ax.set_title("Top 10 busiest routes")
    ax.set_xlabel("Number of listed itineraries")
    fig.tight_layout()
    fig.savefig(plots_dir / "05_top_routes.png", dpi=120)
    plt.close(fig)

    print(f"[plot_flight_eda] saved 5 plots to {plots_dir}/")


# =========================================================================== #
# 2. USER DATA CLEANING + RAW-HISTORY FEATURE MINING
# =========================================================================== #
RE_MONEY = re.compile(r"\$\s?(\d+(?:\.\d+)?)")
RE_HOURS = re.compile(r"(\d+(?:\.\d+)?)\s?(?:hr|hour|hrs|hours)\b")
RE_STOPS = re.compile(r"(\d+)\s?stop")

# Small lexicons for rule-based stance detection. This is intentionally
# simple/transparent (vs. a black-box classifier) so every inferred stance
# can be traced back to the exact phrase that triggered it -- that
# traceability is what Module 3 uses to cite evidence in its explanations.
BUDGET_POSITIVE = ["cheap", "cheapest", "budget", "broke", "steal", "rock-bottom", "value matters", "save"]
BUDGET_NEGATIVE = ["comfort over cost", "not the constraint", "whatever it costs", "the works"]
DIRECT_POSITIVE = ["hate connections", "direct is worth", "direct whenever", "worth paying for"]
DIRECT_NEGATIVE = ["dont care about stops", "don't care about stops", "stops fine", "layover fine", "2 stops fine"]
REDEYE_NEGATIVE = ["redeye", "red-eye", "melt down at night", "kill my mornings"]
FAMILY_KEYWORDS = ["kid", "kids", "stroller", "family"]
LOYALTY_KEYWORDS = ["loyalty", "status", "alliance", "aadvantage", "krisflyer", "skymiles"]


def load_users(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_airline_lookup(df_flights: pd.DataFrame) -> dict:
    """code -> full airline name, used to resolve mentions like 'SQ' or 'JL'
    found inside free-text fragments."""
    return dict(zip(df_flights["airline_code"].dropna(), df_flights["airline_name"].dropna()))


def split_fragments(raw_history: str) -> list[str]:
    """Split a raw_history string into atomic signal fragments.

    This dataset delimits fragments with '|', but the splitter is written
    generically: if no '|' is present it falls back to sentence splitting,
    so the same function still behaves reasonably on messier free text.
    """
    if not isinstance(raw_history, str) or not raw_history.strip():
        return []
    parts = re.split(r"\s*\|\s*", raw_history)
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", raw_history)
    return [p.strip() for p in parts if p.strip()]


def mine_fragment(fragment: str, airline_codes: list[str]) -> dict:
    """Extract structured signals out of one free-text fragment.

    Returns a dict of whatever it found -- callers should treat missing
    keys as 'no signal detected', not as False/0.
    """
    text = fragment.lower()
    signals: dict = {"text": fragment}

    money = RE_MONEY.findall(text)
    hours = RE_HOURS.findall(text)
    if money and hours:
        # e.g. "took a 7hr layover to save $120" -> a revealed price/time
        # trade-off rate for this specific user (see infer_weights below)
        signals["elasticity_usd_per_hour"] = round(float(money[0]) / float(hours[0]), 2)
    elif money:
        signals["mentions_amount_usd"] = float(money[0])
    if hours:
        signals["mentions_hours"] = float(hours[0])

    stops = RE_STOPS.findall(text)
    if stops:
        signals["mentions_stop_count"] = int(stops[0])

    mentioned = [c for c in airline_codes if re.search(rf"\b{re.escape(c)}\b", fragment)]
    if mentioned:
        signals["airlines_mentioned"] = mentioned

    if any(k in text for k in BUDGET_POSITIVE):
        signals["stance_cost"] = "price_sensitive"
    if any(k in text for k in BUDGET_NEGATIVE):
        signals["stance_cost"] = "price_insensitive"

    if any(k in text for k in DIRECT_POSITIVE):
        signals["stance_directness"] = "prefers_direct"
    if any(k in text for k in DIRECT_NEGATIVE):
        signals["stance_directness"] = "connections_ok"

    if any(k in text for k in REDEYE_NEGATIVE):
        signals["avoids_redeye"] = True
    if any(k in text for k in FAMILY_KEYWORDS):
        signals["family_signal"] = True
    if any(k in text for k in LOYALTY_KEYWORDS):
        signals["loyalty_signal"] = True

    return signals


def infer_weights(row: pd.Series, fragment_signals: list[dict]) -> dict:
    """Fuse structured columns (treated as a prior) with mined raw-history
    stances (treated as evidence that can nudge, not override, the prior)
    into a 3-axis multi-objective weight vector for the Module 2 optimizer.
    """
    price_sensitivity_map = {"none": 0.05, "low": 0.2, "medium": 0.5, "high": 0.85}
    direct_pref_map = {"none": 0.1, "moderate": 0.5, "strong": 0.9}

    cost_weight = price_sensitivity_map.get(str(row.get("price_sensitivity", "medium")).lower(), 0.5)
    convenience_weight = direct_pref_map.get(str(row.get("direct_preference", "moderate")).lower(), 0.5)

    stances_cost = [s.get("stance_cost") for s in fragment_signals if "stance_cost" in s]
    if "price_sensitive" in stances_cost:
        cost_weight = min(1.0, cost_weight + 0.15)
    if "price_insensitive" in stances_cost:
        cost_weight = max(0.0, cost_weight - 0.15)

    stances_dir = [s.get("stance_directness") for s in fragment_signals if "stance_directness" in s]
    if "prefers_direct" in stances_dir:
        convenience_weight = min(1.0, convenience_weight + 0.1)
    if "connections_ok" in stances_dir:
        convenience_weight = max(0.0, convenience_weight - 0.1)

    time_weight = max(0.05, round(1.0 - (cost_weight + convenience_weight) / 2, 3))

    total = cost_weight + convenience_weight + time_weight
    weights = {
        "cost_weight": round(cost_weight / total, 3),
        "convenience_weight": round(convenience_weight / total, 3),
        "time_weight": round(time_weight / total, 3),
    }

    elasticities = [s["elasticity_usd_per_hour"] for s in fragment_signals if "elasticity_usd_per_hour" in s]
    if elasticities:
        weights["revealed_usd_per_layover_hour"] = round(float(np.mean(elasticities)), 2)

    return weights


def build_preference_summary(row: pd.Series, fragment_signals: list[dict]) -> str:
    """Compose one natural-language paragraph per user. Module 2 embeds
    this string -- it needs to read like a short bio, not a JSON dump."""
    bits = [
        f"{row.get('trip_purpose', 'mixed')} traveler based in {row.get('home_city', 'unknown city')} "
        f"({row.get('home_airport', '???')}).",
        f"Cabin preference: {row.get('preferred_cabin', 'Economy')}.",
        f"Price sensitivity: {row.get('price_sensitivity', 'medium')}.",
        f"Direct-flight preference: {row.get('direct_preference', 'moderate')} "
        f"(tolerates up to {row.get('max_layover_minutes', 'unspecified')} min layovers).",
        f"Date flexibility: {row.get('date_flexibility_days', 'unspecified')} days.",
        f"Multi-city tendency: {row.get('multi_city_tendency', 'low')}.",
        f"Preferred departure time: {row.get('preferred_departure', 'any')}.",
        f"Baggage needs: {row.get('baggage_preference', 'unspecified')}.",
        f"Seasonal pattern: {row.get('seasonal_pattern', 'unspecified')}.",
    ]
    evidence_texts = [s["text"] for s in fragment_signals][:5]
    if evidence_texts:
        bits.append("Notable history: " + "; ".join(evidence_texts) + ".")
    return " ".join(bits)


def build_user_profiles(df_users: pd.DataFrame, airline_lookup: dict) -> list[dict]:
    airline_codes = list(airline_lookup.keys())
    profiles = []

    for _, row in df_users.iterrows():
        fragments = split_fragments(row.get("raw_history", ""))
        fragment_signals = [mine_fragment(f, airline_codes) for f in fragments]

        weights = infer_weights(row, fragment_signals)
        summary = build_preference_summary(row, fragment_signals)

        profiles.append({
            "user_id": row["user_id"],
            "home_airport": row.get("home_airport"),
            "home_city": row.get("home_city"),
            "preferred_airlines": str(row.get("preferred_airlines", "")).split(";"),
            "preferred_cabin": row.get("preferred_cabin"),
            "price_sensitivity": row.get("price_sensitivity"),
            "direct_preference": row.get("direct_preference"),
            "max_layover_minutes": row.get("max_layover_minutes"),
            "date_flexibility_days": row.get("date_flexibility_days"),
            "trip_purpose": row.get("trip_purpose"),
            "preferred_departure": row.get("preferred_departure"),
            "baggage_preference": row.get("baggage_preference"),
            "seasonal_pattern": row.get("seasonal_pattern"),
            "weights": weights,                 # numeric optimizer input (Module 2)
            "evidence_fragments": fragment_signals,  # traceable mined signals (Module 3 explanations)
            "preference_summary": summary,       # text to embed (Module 2)
        })

    return profiles


# =========================================================================== #
# main
# =========================================================================== #
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading raw data...")
    df_flights_raw = load_flights(FLIGHTS_PATH)
    df_users_raw = load_users(USERS_PATH)
    print(f"  flights: {df_flights_raw.shape}, users: {df_users_raw.shape}")

    print("Cleaning flights...")
    df_flights = clean_flights(df_flights_raw)
    df_flights.to_csv(OUTPUT_DIR / "flights_clean.csv", index=False)
    print(f"  -> saved {len(df_flights)} rows to {OUTPUT_DIR / 'flights_clean.csv'}")

    print("Generating EDA plots...")
    plot_flight_eda(df_flights, PLOTS_DIR)

    print("Mining user preferences from raw_history...")
    airline_lookup = build_airline_lookup(df_flights)
    profiles = build_user_profiles(df_users_raw, airline_lookup)
    with open(OUTPUT_DIR / "user_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2, default=str)
    print(f"  -> saved {len(profiles)} profiles to {OUTPUT_DIR / 'user_profiles.json'}")

    # quick sanity print for one profile
    example = next(p for p in profiles if p["user_id"] == "U02")
    print("\nExample profile (U02):")
    print(f"  weights: {example['weights']}")
    print(f"  summary: {example['preference_summary'][:160]}...")


if __name__ == "__main__":
    main()
