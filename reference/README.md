# Reference configuration

`source-config.json` is the configuration of the earlier experiment whose twelve source heads these studies reuse
as the cached signal (`models.gpt2-medium.source_heads`). Study 1's `code/prepare_inputs.py` reads the source heads
from it; every later step reads them from the studies' own frozen `protocol/config.json`. The file is copied byte for
byte; the studies recorded its SHA-256 among their reference files (`studies/fixed-signal-v1/provenance/PREPARATION-SNAPSHOT.json`).
