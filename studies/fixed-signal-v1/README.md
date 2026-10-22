# AI-08 fixed-signal study — completed 8 October 2026

Read the [completed research report](report/AI-08-COMPLETED-STUDY.md).

The controlled experiment and independent confirmation are finished: **1,408 recipient cases, 80 source pairs and 18,304 condition records**. The directional prediction model reduced confirmation error by **11.00%**, but its absolute improvement, **0.03778**, missed the predeclared **0.05** continuation threshold. The study supports modest context-dependent transformation of a fixed signal; it does not identify a routing mechanism.

Matching source/recipient name order usually helps the intended answer; opposite order usually hurts. Original-batch replay was bitwise identical. A stricter singleton internal-state tolerance failed and is retained, with its measured negligible effect on the sampled prediction scores documented in the report.

- [Completed report and figure](report/AI-08-COMPLETED-STUDY.md).
- [Machine-readable accounting and decision](analysis/STUDY-SUMMARY.json).
- [Reproduction and verification](REPRODUCIBILITY.md).
- `protocol/`: design, analysis rules and timestamped file fingerprints.
- `data/`: fully tokenized pilot and independent confirmation inputs.
- `runs/`: full logits, state arrays, cached signals, condition records and replay evidence.
- `analysis/`: model selection, held-out predictions, controls and audits.
- `provenance/`: original preparation snapshot and unchanged model/reference fingerprints.

The original preparation files at the AI-08 project root are historical and unchanged. This study supplies the implemented runner and completed outcomes. Earlier AI-01 and AI-07 study evidence was not overwritten. No experiment or continuation is running or scheduled for this study.
