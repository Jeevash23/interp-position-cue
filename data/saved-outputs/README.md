# Saved outputs

The studies' results, packed as deterministic tar archives compressed with xz. Member paths are relative to the release root, so `make unpack` (or `python tools/unpack.py`) puts every file where the study code reads it and checks each SHA-256 against `data/PACKED-FILES.json`. Text records inside the archives had personal strings replaced, as listed in `RELEASE-CHANGES.md`.

| Archive | Files | Packed (MB) | Unpacked (MB) | Contents |
|---|---|---|---|---|
| `study1-fixed-signal.tar.xz` | 58 | 1.4 | 16.0 | Study 1: analyses (summaries, behaviour, selection, pairs), evidence audit, run status and summaries (large per-pair arrays not included) |
| `study2-position-routing.tar.xz` | 29 | 2.0 | 15.9 | Study 2: analyses, run status and summaries |
| `study3-pathway-interactions.tar.xz` | 41 | 0.5 | 2.2 | Study 3: screen ranking, selection, predictions, analyses, run status and summaries |

Total packed size: 3.9 MB. Rendered figures are not included; `make figures` draws them from these outputs.
