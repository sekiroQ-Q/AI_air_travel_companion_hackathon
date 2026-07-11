# AI Air Travel Companion -- prototype code

Three modules, meant to run in order. Each is copy-pasteable and runnable
on its own once the previous module's output exists.

```
module1_eda_cleaning.py      -> output/flights_clean.csv, output/user_profiles.json, output/eda_plots/
module2_preference_engine.py -> embeddings + FAISS store + PyTorch multi-objective optimizer
module3_orchestration.py     -> LangChain-orchestrated end-to-end pipeline
```

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn \
            torch faiss-cpu sentence-transformers \
            transformers langchain langchain-community
```

Expected data layout:

```
data/flights_data.csv
data/user_data.csv
```

Run in order:

```bash
python module1_eda_cleaning.py
python module2_preference_engine.py
python module3_orchestration.py
```

## Architecture

`user query -> preference extraction (structured fields + mined raw_history,
fused into numeric weights + a text summary) -> FAISS retrieval (semantic
search over profiles/evidence) -> PyTorch multi-objective optimizer
(hard-constraint filter, then Pareto + weighted scoring across 5 axes)
-> explanation (evidence-cited, trade-off-explicit)`

The optimizer scores five independent axes -- cost, time, stop count,
layover duration, and on-time reliability -- rather than a single
"convenience" number. Earlier versions of this code blended stops/layover
duration/reliability into one hand-tuned formula with fixed coefficients
applied to every user; that's gone. Each axis now gets its own weight
per user, derived in Module 1's `infer_weights()` / `split_convenience_pool()`
from the same evidence-fusion process as the rest of the profile. See
"Notable modeling decision" below for a specific case this surfaced.

Embeddings are used for **retrieval** (finding a user's profile, finding
the raw-history fragments most relevant to a specific query). They are
**not** used to score numeric flight features directly -- that's the
job of the small PyTorch `MultiObjectiveUtility` module operating on
Module 1's normalized price/duration/inconvenience columns. This split
is deliberate: cosine similarity is the wrong tool for "is $640 under
budget," and a deterministic scorer is the wrong tool for "which
raw-history sentence is most relevant to this query."

## Important: offline fallback behavior

This code was built and tested in a sandboxed environment **without
access to huggingface.co**. Rather than hand you untested code for the
embedding and LLM steps, both were built with automatic, honest
fallbacks, and I ran the full pipeline end-to-end against the actual
`flights_data.csv` / `user_data.csv` from your hackathon zip to confirm
everything downstream works:

- **Module 2 embeddings**: tries `sentence-transformers/all-MiniLM-L6-v2`
  first. If the model can't be downloaded (no internet), it automatically
  falls back to a local TF-IDF + SVD embedder -- same interface, same
  FAISS indexing, same retrieval calls, just lower semantic quality
  (lexical overlap instead of true semantic similarity). **On your machine,
  with normal internet access, this fallback will not trigger** -- you'll
  get real MiniLM embeddings automatically the first time you run it.

- **Module 3 query parsing & explanation**: defaults to `llm=None`, which
  uses a zero-dependency rule-based parser and a deterministic template
  explainer. Both were verified to satisfy every item in
  `benchmark_prompts.json`'s `expected_behavior` checklist (destination
  extraction, trade-off with $ and hours, cited evidence, stated
  priorities). To route through a real open-source LLM, uncomment one of
  the two wiring examples at the top of `main()` (Ollama or a local HF
  pipeline) and pass it as `llm=...` -- any failure at call time falls
  back to the rule-based/template path automatically, so a flaky model
  server can't take down your live demo.

I'd recommend keeping the fallback path as your default even once you
have a working LLM -- it's what makes the difference between "the demo
crashed" and "the demo looked slightly less fancy for one query."

## Notable modeling decision: non-monotonic layover preference

Four users in `user_data.csv` (U10, U30, U40, U50) have a raw_history
fragment like *"scared of missing connections, short layovers stress
me"* -- these users are stressed by **short** layovers, not long ones.
That's the opposite of what the rest of the pipeline assumes (`layover_hours`
is scored as "less is always better"), and a linear utility function
cannot represent a preference that's bad at both extremes and fine in
the middle.

Rather than silently score these four users wrong, `split_convenience_pool()`
detects this pattern and zeroes out their `layover_weight` entirely,
shifting that share to `reliability_weight` instead -- an unreliable
connecting flight is the actual mechanism behind "missing the
connection," so it's the closest available proxy to their real concern.
`explain_template()` also surfaces this as an explicit note in the
output rather than hiding it. The fully correct fix -- a U-shaped
penalty around an ideal buffer time, or a hard minimum-connection-time
constraint -- is real additional work and is listed under Future
improvements rather than attempted here.

## Assumptions

- `origin` in a query defaults to the user's `home_airport`; an explicit
  origin in the query text would need one extra parsing rule (not added,
  since every benchmark prompt implies traveling from home).
- An explicit constraint stated in the live query (e.g. "hate layovers")
  overrides the stored profile for that query -- stored preferences are a
  prior, not a hard rule the user can't contradict in the moment.
- Currency is assumed USD throughout (true for the provided dataset);
  `clean_flights()` warns rather than silently mis-converting if a
  non-USD row ever appears.
- Multi-city routing (chaining several origin/destination legs, or
  solving flexible city ordering) is not implemented in this pass --
  `filter_candidates` handles single-destination and simple
  multi-destination-list queries. See Limitations.
- `split_convenience_pool()`'s default 55/35/10 split across
  stops/layover/reliability is a reasoned prior, not fit to any ground
  truth -- there's no labeled data in this dataset to fit it against.
  Evidence-based nudges shift the split per user; the starting point
  itself is a judgment call, stated here rather than hidden in a
  formula (see the previous conversation turn for the fuller reasoning).

## Limitations

- Multi-city trip ordering (e.g. "London + Paris + Rome, best order") is
  not implemented -- `score_and_rank` scores each origin/destination pair
  independently, not a chained itinerary. This is the highest-value next
  addition if your problem statement scope includes it (see Future
  improvements).
- Date-window search is not implemented -- Module 1/2/3 filter by route
  and constraints but do not yet search across a flexible date range
  (`date_flexibility_days` is extracted but not yet applied as a filter).
- The Pareto frontier computation is O(n^2); fine at single-route
  candidate-set scale (tens to low hundreds of rows) but would need a
  sweep-line algorithm if used on unfiltered data. It now runs across 5
  axes instead of 3, so expect a larger non-dominated set than before --
  that's a correct consequence of no longer collapsing independent
  trade-offs into one number, not a regression.
- The "wants a longer buffer" workaround above (zeroing `layover_weight`,
  boosting `reliability_weight`) is a reasonable approximation, not a
  real fix for non-monotonic preferences -- it still can't express "a
  40-minute connection is worse than a 90-minute one, which is better
  than a 9-hour one."
- `MultiObjectiveUtility`'s weights are set from the fused profile at
  inference time and not trained from any real feedback signal -- there
  is no click/booking data in this dataset to train against.

## Future improvements

- **Multi-city routing**: model the 35 airports as a graph and solve
  city ordering with OR-Tools' routing solver (small enough problem size
  -- typically <=5 cities -- for brute-force ordering too).
- **Date-flexibility search**: sweep `date_flexibility_days` around the
  requested date and surface the cheapest date in the window alongside
  the requested-date recommendation.
- **A real non-monotonic layover penalty**: replace the zero-weight
  workaround with a proper U-shaped penalty (or a hard minimum-connection
  constraint pulled from the "wants_buffer" evidence) so those four users'
  actual preference -- a buffer, not a minimum -- gets modeled directly
  instead of approximated via the reliability axis.
- **Preference-weight fine-tuning**: `MultiObjectiveUtility`'s weights are
  `nn.Parameter`s specifically so they're gradient-ready. With real
  booking/click feedback, a pairwise preference loss (or a lightweight
  contextual bandit) could refine per-user weights online -- this is the
  natural place a reinforcement-learning-style approach would plug in,
  deliberately deferred rather than built on synthetic reward signals.
- **Swap the offline TF-IDF fallback for a fine-tuned encoder**: if
  demo-day connectivity is uncertain, consider vendoring the MiniLM
  weights ahead of time (`SentenceTransformer(...).save(...)`) rather
  than relying on a live download.