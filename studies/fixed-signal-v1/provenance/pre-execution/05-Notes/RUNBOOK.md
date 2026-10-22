# Starting the context dependent signal pilot

Preparation is limited to text/configuration processing. The next experimental stage begins only after the user resumes work and shared resources are available.

## Safe commands now

Run `make check` to validate prompt role relationships, split separation, condition counts and fixed-source reuse. `make plan` prints counts. Neither loads tokenizer/model libraries, reads weight files, queries the GPU, creates workers or installs anything.

## Next work

1. Confirm the latest draft and literature boundaries. Read AI-01's preservation instructions.
2. Tokenize candidate prompts with the local retained tokenizer and document failures by complete matched block. Revise and record changes before scientific freeze.
3. Implement an isolated inference adapter following [the execution contract](../02-Code/EXECUTION-CONTRACT.md). Validate untouched/zero-write equivalence, source hashes and semantic-position alignment on a tiny development batch.
4. Measure memory and time, then qualify clean competence and source separation. No present estimate establishes runtime for the new adapter.
5. Implement the two predictive analyses and verify them using known gain-only and context-transform synthetic arrays before model interpretation. This verification is not a model experiment and may be done later without a GPU.
6. Freeze qualified inputs, source/recipient mapping, numerical tolerances, fitting choices and decision thresholds before holdout results. Run once in a fresh output directory and retain all failures.
7. Decide whether mediator experiments are justified; do not automatically start a head sweep.

## Resource coordination

Existing model and source files are references, not targets for editing. Do not upgrade AI-01's environment or launch its original pipelines. No weights are copied into this project. If the retained environment is unsuitable, create an isolated environment later with recorded exact versions.

Before any future GPU allocation use the existing portfolio lock at `<gpu-lock>` relative to the project root. Acquire it nonblocking and defer if busy. A CPU fallback also needs a resource check. Do not terminate other jobs, create a replacement lock, or schedule automatic retries. No experimental execution is included in the current request.
