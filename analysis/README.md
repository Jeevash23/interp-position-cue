# Analyses across studies and after the studies

Everything here reads saved outputs of the sealed studies and writes only to `results/`. No model is run. The
analyses were added after the studies ended and are labelled **post hoc** in the manuscript.

| Script | Output | What it does |
|---|---|---|
| `posthoc_intervals.py` | `results/posthoc-intervals.json` | Intervals for quantities that the frozen analyses report without one (Study 1's relative improvement and order split; Study 2's clamps of keys, values, attention, outputs and other heads, foreign-source ratios and benefit/harm shares; Study 3's benefit/harm shares) and the order contrast by wording. Each study's own resampling scheme, draw count and seed; the script first reproduces saved intervals exactly and stops if it cannot |
| `posthoc_query_feed.py` | `results/query-feed.json`, `results/query-feed-development.json` (`--bank development`) | From Study 3's saved full final-position traces (16 recipients per bank, chosen before measurement): how much of the order-dependent change in the five name-mover queries survives when heads 15.4 and 12.3, or MLP block 14, are held at their unedited values, and how much their transplant recreates. Descriptive, no intervals; includes the check that head 12.3, which precedes every query head, is unaffected by holding the queries |
| `replay_study_analyses.py` | `results/replay-check.json` | Reruns each study's documented saved-output analysis in a temporary folder and compares every file with the saved analysis |

Run them with `make -j1 analysis` and `make -j1 replay` from the project folder.
