# Inference adapter contract

The implemented code prepares text and condition plans only. The new model adapter has not been written or numerically qualified. Existing AI-01 code is a reference and must remain unchanged.

## Qualified inputs

Resolve model/tokenizer revisions and paths locally, with network access disabled in the future loader. Map semantic repeated-subject and answer positions through tokenization. Store all token IDs and mappings with hashes in the run inputs. Do not infer token indices from string length or transfer absolute indices across wordings without checking them.

Cache each declared source clean/corrupted pair once, deriving the correction exactly as specified in the protocol. For a fixed source ID, the vector bytes and layer/head layout must remain unchanged across recipient cells. Record the signal hash in every output row. Recipient self corrections are separate controls, not substitute fixed sources.

## Execution and records

Require an explicit future execution command, a qualified configuration and a fresh run directory. Acquire shared-resource coordination before importing/allocating a GPU model. Refuse downloads, existing output collisions and incompatible resumes. Write per-batch completion state atomically.

Each record includes run ID, prompt group/split, source ID, source hash, recipient ID, condition, strength, random draw ID, token-position mapping, original and patched predictions, margins, response-array reference, failure reason, configuration hash, environment versions and timing. Report source-cache and calibration costs separately from the 3328 planned condition evaluations.

## Numerical gates

Establish untouched versus zero-write equivalence, self correction semantics, deterministic cached-source reuse, and identical results across representative batch boundaries. Record absolute/relative tolerances justified by dtype and observed repeat error. These checks must be run later; static preparation cannot establish them.

## Response analysis

Fit dimensionality reduction, rank and regularization on development groups only. Preserve raw original-space responses and model errors. Holdout reference responses are permitted only as the explicit, costed inputs described in the protocol. Keep zero-norm/near-noise responses and unsuccessful clean prompts in coverage accounting.
