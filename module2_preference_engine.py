"""
Module 2 - Core AI Model: Preference Extraction & Optimization
================================================================
AI Air Travel Companion (hackathon prototype)

Responsibilities:
  1. Embed each user's `preference_summary` (from Module 1) with a
     lightweight sentence-transformer, so preferences can be retrieved
     semantically rather than by exact user_id lookup alone.
  2. Store those embeddings in a local FAISS index (no server required).
  3. Score candidate flights against a user's fused profile with a small
     PyTorch multi-objective utility function (price vs. duration vs.
     inconvenience), after hard constraint filtering.

Design note on embeddings vs. the optimizer (read this before you skim
past it): embeddings are used for *semantic retrieval* -- finding a
user's profile, and finding which raw-history fragments are most relevant
to a specific query, for explanation grounding. They are NOT used to
directly score numeric flight features like price or duration -- cosine
similarity between a preference-summary embedding and a flight-price
number is not a meaningful operation. The actual multi-objective scoring
is done by a small, deterministic-at-inference PyTorch module operating
on the normalized numeric features Module 1 already engineered. This
hybrid (embeddings for retrieval + evidence, a scorer for ranking) is
more robust and more explainable than trying to force one tool to do
both jobs.

Offline fallback: this environment may not have internet access to
huggingface.co at build/demo time. `PreferenceEmbedder` tries to load a
real sentence-transformer first and automatically falls back to a local
TF-IDF+SVD embedder if that fails, so the pipeline still runs end to end.
On a machine with normal internet access the fallback branch simply never
triggers.

Run:
    python module2_preference_engine.py
(expects module1's output/flights_clean.csv and output/user_profiles.json
to already exist)
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

OUTPUT_DIR = Path("output")
FLIGHTS_CLEAN_PATH = OUTPUT_DIR / "flights_clean.csv"
PROFILES_PATH = OUTPUT_DIR / "user_profiles.json"


# =========================================================================== #
# 1. TRANSFORMER-BASED PREFERENCE EMBEDDER (with offline fallback)
# =========================================================================== #
class PreferenceEmbedder:
    """Wraps a small HuggingFace sentence-transformer for embedding
    preference summaries. Swap EMBED_MODEL_NAME for any
    sentence-transformers-compatible small BERT variant."""

    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, ~90MB
    FALLBACK_DIM = 128

    def __init__(self, device: str | None = None):
        self.mode = None
        self._model = None
        self._tfidf = None
        self._svd = None
        # auto-detect GPU; explicit device= overrides. At 50 profiles this
        # model is fast either way -- GPU only starts to matter if you
        # scale this up to embedding thousands of flight descriptions too.
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_backend()

    def _init_backend(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.EMBED_MODEL_NAME, device=self.device)
            self.mode = "transformer"
            print(f"[PreferenceEmbedder] loaded {self.EMBED_MODEL_NAME} on {self.device}")
        except Exception as e:
            print(
                f"[PreferenceEmbedder] could not reach huggingface.co ({e.__class__.__name__}) "
                f"-- falling back to a local TF-IDF+SVD embedder so the pipeline keeps running "
                f"offline. This is a *quality* fallback, not a functional one: FAISS indexing, "
                f"retrieval, and the optimizer below all behave identically either way. On a "
                f"machine with internet access this branch will not trigger."
            )
            self.mode = "tfidf"

    def fit_fallback(self, corpus: list[str]):
        """Only relevant in tfidf mode: fit the vectorizer once on the full
        profile corpus before encoding anything."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self._tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
        X = self._tfidf.fit_transform(corpus)
        n_components = max(2, min(self.FALLBACK_DIM, X.shape[0] - 1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(X)

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.mode == "transformer":
            emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(emb, dtype="float32")

        if self._tfidf is None:
            raise RuntimeError("call fit_fallback(corpus) once before encode() in tfidf mode")
        X = self._tfidf.transform(texts)
        emb = self._svd.transform(X).astype("float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms


# =========================================================================== #
# 2. FAISS VECTOR STORE
# =========================================================================== #
class UserPreferenceStore:
    """Local FAISS index over user preference-summary embeddings, plus the
    profile metadata FAISS itself doesn't store."""

    def __init__(self, embedder: PreferenceEmbedder):
        self.embedder = embedder
        self.index = None
        self.dim = None
        self.user_ids: list[str] = []
        self.metadata: dict[str, dict] = {}

    def build(self, profiles: list[dict]):
        self.user_ids = [p["user_id"] for p in profiles]
        self.metadata = {p["user_id"]: p for p in profiles}
        summaries = [p["preference_summary"] for p in profiles]

        if self.embedder.mode == "tfidf":
            self.embedder.fit_fallback(summaries)

        embeddings = self.embedder.encode(summaries)
        self.dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)  # inner product on normalized vecs == cosine sim
        self.index.add(embeddings)
        print(f"[UserPreferenceStore] indexed {len(self.user_ids)} users (dim={self.dim}, mode={self.embedder.mode})")

    def get_profile(self, user_id: str) -> dict:
        if user_id not in self.metadata:
            raise KeyError(f"unknown user_id: {user_id!r}")
        return self.metadata[user_id]

    def search_similar_users(self, query_text: str, k: int = 3) -> list[dict]:
        """Semantic retrieval demo: find the k users whose preference
        summary most resembles a free-text description."""
        q_emb = self.embedder.encode([query_text])
        scores, idxs = self.index.search(q_emb, k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            uid = self.user_ids[idx]
            out.append({"user_id": uid, "similarity": float(score)})
        return out

    def evidence_for_query(self, user_id: str, query_text: str, k: int = 3) -> list[str]:
        """Retrieve the k raw-history fragments (belonging to this user)
        most semantically relevant to the current query. Module 3 uses
        this to ground its explanation instead of dumping full history."""
        profile = self.get_profile(user_id)
        fragments = [f["text"] for f in profile.get("evidence_fragments", []) if "text" in f]
        if not fragments:
            return []
        frag_emb = self.embedder.encode(fragments)
        q_emb = self.embedder.encode([query_text])[0]
        sims = frag_emb @ q_emb
        top_idx = np.argsort(-sims)[:k]
        return [fragments[i] for i in top_idx]


# =========================================================================== #
# 3. PYTORCH MULTI-OBJECTIVE UTILITY FUNCTION
# =========================================================================== #
class MultiObjectiveUtility(nn.Module):
    """A small, differentiable multi-objective utility function.

    Five axes -- price, duration, stops, layover duration, on-time
    unreliability -- normalized to [0, 1] *per query* (see
    normalize_candidates() below), not against the global dataset. Weights
    are initialized from the fused profile (Module 1's `weights` field)
    and pushed through softmax so they stay positive and sum to 1. Utility
    is the negative weighted sum: lower cost/time/stops/layover/
    unreliability => higher utility.

    Stops, layover duration, and on-time reliability used to be
    pre-blended into a single hand-tuned "inconvenience_score" formula
    with fixed global coefficients. They're independent axes here, each
    with its own per-user weight from Module 1's split_convenience_pool()
    -- so a user who's revealed they mind connection count but not
    layover length (or vice versa) is actually scored differently,
    instead of both being flattened into one number.

    This is deliberately a real nn.Module with learnable parameters, not
    a plain dict lookup: it's a genuine (if small) PyTorch component, and
    it sets up the natural future-work extension -- fine-tuning these
    weights against real booking/click feedback via a pairwise preference
    loss is a few lines from here, without touching anything upstream.
    """

    AXES = ("cost", "time", "stops", "layover", "reliability")
    RAW_COLS = ("price", "duration_minutes", "stops", "layover_hours", "reliability_penalty")

    def __init__(self, weights: dict):
        super().__init__()
        vals = [weights.get(f"{axis}_weight", 0.2) for axis in self.AXES]
        init = torch.log(torch.tensor(vals, dtype=torch.float32) + 1e-6)
        self.raw_weights = nn.Parameter(init)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [N, 5] columns, in RAW_COLS order, each already
        normalized to [0, 1] *within this candidate set* by
        normalize_candidates()."""
        w = self.weights()
        return -(features * w).sum(dim=1)


def normalize_candidates(df_candidates: pd.DataFrame, cols=MultiObjectiveUtility.RAW_COLS) -> pd.DataFrame:
    """Min-max normalize each raw axis *within this specific candidate
    set*, not against the whole dataset.

    This replaces normalizing once in Module 1 against the global min/max
    across all 50,000 flights. Global normalization meant a user's stated
    weight (e.g. "cost matters 30%") swung the ranking by a different,
    route-dependent amount every time -- e.g. CPT->DOH's price only spans
    ~$442-$4,566 against a global range of ~$40-$17,832, so price_norm for
    that route only ever occupied about a quarter of [0, 1] while
    duration_norm occupied nearly the whole range, silently discounting
    cost's actual influence to a fraction of what the weight implied.
    Normalizing per-query instead makes each axis span its full local
    range, so the weights behave consistently regardless of which route
    is queried.

    Trade-off worth knowing: this is a genuine trade, not a free fix. A
    candidate set with very little real-world spread (e.g. only 2-3
    flights on a thin route, a few dollars apart) will still get stretched
    across the full [0, 1] range on that axis -- a trivial price
    difference can look as decisive as a $2,000 one would on a busier
    route. Global normalization protected against that at the cost of the
    cross-route inconsistency above; this pipeline takes the opposite
    trade-off deliberately, since "the weights should mean what they say
    for this query" matters more here than "small dollar gaps on thin
    routes shouldn't look dramatic."
    """
    normed = {}
    for col in cols:
        cmin, cmax = df_candidates[col].min(), df_candidates[col].max()
        if cmax > cmin:
            normed[f"{col}_norm"] = (df_candidates[col] - cmin) / (cmax - cmin)
        else:
            # one candidate, or every candidate tied on this axis -- there's
            # nothing to differentiate, so it contributes 0 either way
            normed[f"{col}_norm"] = pd.Series(0.0, index=df_candidates.index)
    return pd.DataFrame(normed, index=df_candidates.index)


# =========================================================================== #
# 4. CONSTRAINT FILTERING + PARETO FRONTIER
# =========================================================================== #
def filter_candidates(
    df_flights: pd.DataFrame,
    origin: str,
    destinations: str | list[str],
    cabin: str | None = None,
    max_layover_minutes: int | None = None,
    min_seats: int = 1,
) -> pd.DataFrame:
    """Hard constraints, applied before any scoring happens."""
    df = df_flights[df_flights["origin"] == origin]
    if isinstance(destinations, str):
        destinations = [destinations]
    df = df[df["destination"].isin(destinations)]
    if cabin:
        df = df[df["cabin_class"] == cabin]
    if max_layover_minutes is not None:
        df = df[df["layover_minutes"] <= max_layover_minutes]
    df = df[df["seats_available"] >= min_seats]
    return df.reset_index(drop=True)


def pareto_mask(df: pd.DataFrame, cols=("price", "duration_minutes", "stops", "layover_hours", "reliability_penalty")) -> np.ndarray:
    """Boolean mask marking Pareto-optimal (non-dominated) rows across the
    given minimize-columns. Normalization is monotonic, so running this on
    raw columns vs. the _norm columns gives identical results -- raw units
    are used here only because they read better if you print df_scored
    for debugging. O(n^2) -- fine at the scale of a filtered single-route
    candidate set (dozens to low hundreds of rows). Swap for a sweep-line
    algorithm if you ever filter down to thousands of rows.

    Five axes (vs. the original three) means fewer rows dominate each
    other, so expect a larger frontier than before -- that's a correct
    consequence of tracking real independent trade-offs, not a bug. If
    you want a smaller frontier for a cleaner demo slide, cap it in the
    caller (e.g. take the cheapest, fastest, and top-utility pick) rather
    than collapsing axes back together here.
    """
    values = df[list(cols)].to_numpy()
    n = len(values)
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[i]:
                continue
            if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                dominated[i] = True
    return ~dominated


# =========================================================================== #
# 5. SCORE + RANK (the "optimization engine")
# =========================================================================== #
def score_and_rank(df_candidates: pd.DataFrame, profile: dict, top_k: int = 3) -> dict:
    """Scores hard-constraint-filtered candidates with the PyTorch utility
    function and returns the top-k plus the full Pareto frontier, which
    Module 3 uses to build explicit trade-off explanations."""
    if df_candidates.empty:
        return {"top_k": [], "pareto_frontier": [], "learned_weights": None}

    weights = profile.get("weights", {})
    utility_fn = MultiObjectiveUtility(weights)

    feat_cols = ["price_norm", "duration_minutes_norm", "stops_norm", "layover_hours_norm", "reliability_penalty_norm"]
    features = torch.tensor(df_candidates[feat_cols].to_numpy(), dtype=torch.float32)

    with torch.no_grad():
        utility = utility_fn(features).numpy()

    df_scored = df_candidates.copy()
    df_scored["utility_score"] = utility
    df_scored["is_pareto_optimal"] = pareto_mask(df_scored)

    ranked = df_scored.sort_values("utility_score", ascending=False)
    top_k_rows = ranked.head(top_k)
    pareto_rows = df_scored[df_scored["is_pareto_optimal"]].sort_values("price")

    keep_cols = [c for c in [
        "flight_id", "airline_name", "origin", "destination", "departure_utc",
        "price", "duration_hours", "stops", "layover_hours", "cabin_class",
        "on_time_performance", "utility_score", "is_pareto_optimal",
    ] if c in df_scored.columns]

    learned = utility_fn.weights().detach().numpy().tolist()
    return {
        "top_k": top_k_rows[keep_cols].to_dict(orient="records"),
        "pareto_frontier": pareto_rows[keep_cols].to_dict(orient="records"),
        "learned_weights": {axis: round(v, 3) for axis, v in zip(MultiObjectiveUtility.AXES, learned)},
    }


# =========================================================================== #
# main / smoke test
# =========================================================================== #
def main():
    print("Loading Module 1 outputs...")
    df_flights = pd.read_csv(FLIGHTS_CLEAN_PATH, parse_dates=["departure_utc", "arrival_utc"])
    with open(PROFILES_PATH) as f:
        profiles = json.load(f)
    print(f"  flights: {df_flights.shape}, profiles: {len(profiles)}")

    embedder = PreferenceEmbedder()
    store = UserPreferenceStore(embedder)
    store.build(profiles)

    # --- demo: semantic retrieval -----------------------------------------
    print("\nSemantic search: 'traveler who avoids layovers and pays for comfort'")
    for r in store.search_similar_users("traveler who avoids layovers and pays for comfort", k=3):
        print(f"  {r['user_id']}  similarity={r['similarity']:.3f}")

    # --- demo: constrained multi-objective optimization --------------------
    user_id = "U01"
    profile = store.get_profile(user_id)
    origin = profile["home_airport"]
    # pick a destination that actually has inventory from this origin
    candidates_any = df_flights[df_flights["origin"] == origin]["destination"].value_counts()
    destination = candidates_any.index[0]

    print(f"\nOptimizing for {user_id} ({origin} -> {destination})...")
    candidates = filter_candidates(
        df_flights, origin=origin, destinations=destination,
        max_layover_minutes=profile.get("max_layover_minutes"),
    )
    result = score_and_rank(candidates, profile, top_k=3)

    print(f"  candidates after constraints: {len(candidates)}")
    print(f"  learned utility weights: {result['learned_weights']}")
    print("  top pick:")
    if result["top_k"]:
        top = result["top_k"][0]
        print(f"    {top['airline_name']} {origin}->{destination}: "
              f"${top['price']:.0f}, {top['duration_hours']:.1f}h, {top['stops']} stops")
    print(f"  pareto frontier size: {len(result['pareto_frontier'])}")


if __name__ == "__main__":
    main()