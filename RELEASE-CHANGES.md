# Changes from the original project

This release was exported from the archived project folder in which the studies ran. Files are copied byte for
byte except those listed below. There are three kinds of change:

- **adapted:** code whose paths pointed into other folders on the original machine (the model checkpoint, the
  shared GPU lock, the earlier experiment's configuration and prompt folders) now uses `MODELS_DIR`, `GPU_LOCK`,
  `reference/` or `EARLIER_PROJECT_DIR`. The reasons column says which.
- **redacted:** records in which personal or machine-specific strings were replaced: absolute paths and other
  projects' folder names by placeholders (`<project>`, `<portfolio>`, `<portfolio-root>`, `<workspace>`, `<models>`,
  `<environment>`, `<earlier-project>`, `<sibling-project>`, `<gpu-lock>`, `<home>`) and the names of the writing and coding tools in working records by the placeholder `[redacted]` (the assistants used are named in README.md, under "Use of AI assistants"). Nothing else in these files changed; the counts show how many strings of each
  kind were replaced.
- **not shipped:** files named in a freeze record but left out: raw run outputs, large saved arrays, the studies'
  written reports and figures, and two working notes.

Inputs larger than 1 MB are shipped compressed (`.xz`) and saved outputs as archives under
`data/saved-outputs/`; `tools/unpack.py` restores them byte for byte and checks each SHA-256 against
`data/PACKED-FILES.json`. Compression is not a change and is not listed below.

To check every freeze record against this release: `make check-freezes`.

## Files that differ from the project originals

| File | Change | Original SHA-256 | Frozen in |
|---|---|---|---|
| `provenance/preparation-2026-10-07/02-Code/support.py` | redacted (time zone x1) | `b405f60393c511f6d096fbe0d7007a05e17b4ddee5f0980d33c5be58734900d4` | - |
| `provenance/preparation-2026-10-07/03-Data/prepared/preparation-manifest.json` | redacted (time zone x1) | `a06d9752fb549171c0366fceaa8b8f1a4987ace63c023dc117c2417b4ac7ed4d` | - |
| `provenance/preparation-2026-10-07/05-Notes/RUNBOOK.md` | redacted (machine path x1) | `557bcfaced457e2b5a7a33ce76bc0ad147d58679ad8dc59002215235d73ea1e6` | - |
| `provenance/preparation-2026-10-07/06-Study-History/ASSET-REFERENCES.json` | redacted (sibling project x4) | `5b354b6343184b034ff1d26e6e0df69191fe1c1631ff737954bf3cdae8b40143` | - |
| `provenance/preparation-2026-10-07/06-Study-History/PREPARATION-CHECK.json` | redacted (time zone x1) | `7f3148b1ed287b4a206ea55c603b53d73dc8b14805b05b8a0e3df25875755b21` | - |
| `provenance/preparation-2026-10-07/06-Study-History/STATUS.json` | redacted (time zone x1) | `f81fd6c2a4127dd508a90a709f0b8aba4ca86bcc59c8492c73d8bc30af9fb2d1` | - |
| `provenance/preparation-2026-10-07/README.md` | redacted (sibling project x1) | `3f7c8adc7374e8ce20f057b76d5932a07952c334bcee02242a2b7e47f30aee6a` | - |
| `studies/fixed-signal-v1/REPRODUCIBILITY.md` | redacted (environment path x2, model path x1, portfolio folder x1, project folder x2) | `c65f2b983c32a446a63207148690f55bbc8aca383c698314c839748832e5bbb1` | - |
| `studies/fixed-signal-v1/analysis/EVIDENCE-AUDIT.json` | redacted (sibling project x4); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `0031c4860094adbb7a26c15f86d5b733097ace82c7d4bb454ad998592f176f80` | - |
| `studies/fixed-signal-v1/code/core.py` | adapted: the checkpoint comes from MODELS_DIR (default models/ in the release) and the GPU lock from GPU_LOCK | `f1c300f7f69cb65002f9a7f10e5c2b49ff75f60203c3fa3b03de939b0c464654` | `studies/fixed-signal-v1/protocol/QUALIFICATION-FREEZE.json` |
| `studies/fixed-signal-v1/code/prepare_confirmation.py` | adapted: the earlier project's folder comes from EARLIER_PROJECT_DIR | `9ba31abc921d972cbd296d48731bde4b491e7bc2f6eb5b05b5a4ab689cabb9d7` | `studies/fixed-signal-v1/protocol/CONFIRMATION-DESIGN-FREEZE.json` |
| `studies/fixed-signal-v1/code/prepare_inputs.py` | adapted: the source-head configuration is read from the copy in reference/ | `1ae3de2a33a2f5ef8e2ddad2fd40889ff26b340ef5752c56d547115d378dadd9` | `studies/fixed-signal-v1/protocol/QUALIFICATION-FREEZE.json` |
| `studies/fixed-signal-v1/data/confirmation-qualified.json` | redacted (sibling project x98); shipped inside `studies/fixed-signal-v1/data/confirmation-qualified.json.xz` | `0bf66eaf210f78effaebd1b9654788c282c2480fefa7278c0d304132afd6c387` | `studies/fixed-signal-v1/protocol/HELDOUT-FREEZE.json` |
| `studies/fixed-signal-v1/provenance/MODEL-ASSETS.json` | redacted (absolute path x1, model path x1) | `886a32034b3a2bb2546caddbbc72bac25adaef079c15a2e98d06916a2f58a5a6` | `studies/fixed-signal-v1/protocol/RESPONSE-FREEZE.json` |
| `studies/fixed-signal-v1/provenance/PREPARATION-SNAPSHOT.json` | redacted (sibling project x4) | `c213ec30c98f8190fef3b463afe315c8628529a6a7d297870d84741a60090cf1` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/02-Code/support.py` | redacted (time zone x1) | `b405f60393c511f6d096fbe0d7007a05e17b4ddee5f0980d33c5be58734900d4` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/03-Data/prepared/preparation-manifest.json` | redacted (time zone x1) | `a06d9752fb549171c0366fceaa8b8f1a4987ace63c023dc117c2417b4ac7ed4d` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/05-Notes/RUNBOOK.md` | redacted (machine path x1) | `557bcfaced457e2b5a7a33ce76bc0ad147d58679ad8dc59002215235d73ea1e6` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/06-Study-History/ASSET-REFERENCES.json` | redacted (sibling project x4) | `5b354b6343184b034ff1d26e6e0df69191fe1c1631ff737954bf3cdae8b40143` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/06-Study-History/PREPARATION-CHECK.json` | redacted (time zone x1) | `7f3148b1ed287b4a206ea55c603b53d73dc8b14805b05b8a0e3df25875755b21` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/06-Study-History/STATUS.json` | redacted (time zone x1) | `f81fd6c2a4127dd508a90a709f0b8aba4ca86bcc59c8492c73d8bc30af9fb2d1` | - |
| `studies/fixed-signal-v1/provenance/pre-execution/README.md` | redacted (sibling project x1) | `3f7c8adc7374e8ce20f057b76d5932a07952c334bcee02242a2b7e47f30aee6a` | - |
| `studies/fixed-signal-v1/runs/batch-replay/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `9fcfac25eb81e52481d174a8890f18b5b4c6de3033b1c806814ba85ff865f97c` | - |
| `studies/fixed-signal-v1/runs/confirmation/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `bc23cc804c8bc1fa3b05d997190be018ec86b170a6b01d414035bebeaaf4b79f` | - |
| `studies/fixed-signal-v1/runs/development/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `8f6cee4a25bc820bbbc7fda09bb3fd95bfbd4b44ea6c905199ac962dcb60b1f0` | - |
| `studies/fixed-signal-v1/runs/holdout/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `25605bcbf2f648508112e772ea8de4f15a3c202832afc60de677c74226175436` | - |
| `studies/fixed-signal-v1/runs/model-replay/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `ba815e9ad578f18dc16d3853a758a71b85d61ff0314fe536193e787cb3d941c7` | - |
| `studies/fixed-signal-v1/runs/qualification/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study1-fixed-signal.tar.xz` | `42c6a5d77dff4cb8cc44ff06410b2b2256c2e6dfc6c67094793e2ab05ba67f27` | `studies/fixed-signal-v1/protocol/RESPONSE-FREEZE.json` |
| `studies/pathway-interactions-v3/REPRODUCIBILITY.md` | redacted (environment path x2, machine details x1, portfolio folder x1, project folder x2) | `5866d4d3ffd68aecbadcc6ce9e1b1bf8cc595b2d454e7cc2896a6085397d464e` | - |
| `studies/pathway-interactions-v3/code/core.py` | adapted: the checkpoint comes from MODELS_DIR (default models/ in the release) and the GPU lock from GPU_LOCK | `f1c300f7f69cb65002f9a7f10e5c2b49ff75f60203c3fa3b03de939b0c464654` | `studies/pathway-interactions-v3/protocol/DESIGN-FREEZE.json` |
| `studies/pathway-interactions-v3/provenance/MODEL-ASSETS.json` | redacted (absolute path x1, model path x1) | `886a32034b3a2bb2546caddbbc72bac25adaef079c15a2e98d06916a2f58a5a6` | - |
| `studies/pathway-interactions-v3/runs/confirmation/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study3-pathway-interactions.tar.xz` | `078d3fd0a89426052a1fc502377101f4d1eb89c3aa310d715c2f261c0acf4bda` | - |
| `studies/pathway-interactions-v3/runs/development/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study3-pathway-interactions.tar.xz` | `e0c517ef025245cefd8462e79df4bfd0d311d5a758cdbeeee81f9ba2211b17b2` | - |
| `studies/pathway-interactions-v3/runs/qualification/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study3-pathway-interactions.tar.xz` | `76bf9694206e300773d2222d3eef09532c8dd5c9c780a6a853291255a69f1f1d` | `studies/pathway-interactions-v3/protocol/SELECTION-FREEZE.json` |
| `studies/pathway-interactions-v3/runs/replay/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study3-pathway-interactions.tar.xz` | `450763eb7816c4245eefa712597aac143ac6b22b0e2e2329d66ebfb48ae6d5dd` | - |
| `studies/pathway-interactions-v3/runs/screen/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study3-pathway-interactions.tar.xz` | `0aa652c12b9a698d6f89811c6f29ebeebb2f0afcbefde661707546e965dde3d3` | `studies/pathway-interactions-v3/protocol/SELECTION-FREEZE.json` |
| `studies/position-routing-v2/REPRODUCIBILITY.md` | redacted (environment path x2, machine details x1, model path x1, portfolio folder x1, project folder x2) | `82e99053105609e5ab5917c9a0ba74b8383c012382f357358e4936061751a613` | - |
| `studies/position-routing-v2/analysis/STUDY-SUMMARY.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study2-position-routing.tar.xz` | `21108a4481841ac7d70c5eae4f54b27a386d6ebc3963294d4af8856fc5fbc77c` | - |
| `studies/position-routing-v2/code/core.py` | adapted: the checkpoint comes from MODELS_DIR (default models/ in the release) and the GPU lock from GPU_LOCK | `f1c300f7f69cb65002f9a7f10e5c2b49ff75f60203c3fa3b03de939b0c464654` | `studies/position-routing-v2/protocol/CAUSAL-FREEZE.json`, `studies/position-routing-v2/protocol/QUALIFICATION-FREEZE.json` |
| `studies/position-routing-v2/code/summarize_study.py` | redacted (machine details x1) | `82c2461e926626ba656599040fd9fe023bbd6d040ead219f3c70ba3f81dbc15c` | - |
| `studies/position-routing-v2/provenance/MODEL-ASSETS.json` | redacted (absolute path x1, model path x1) | `886a32034b3a2bb2546caddbbc72bac25adaef079c15a2e98d06916a2f58a5a6` | - |
| `studies/position-routing-v2/provenance/PREVIOUS-STUDY.json` | redacted (absolute path x1) | `d826e5333b6c4990f6ef5c44012e3610e9802e76012bc8a7afa8fe0b950a6181` | - |
| `studies/position-routing-v2/runs/confirmation/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study2-position-routing.tar.xz` | `39eafd848be7bcc08f42003eab38b5c522a97847e72d517ef816f49e816051b0` | - |
| `studies/position-routing-v2/runs/development/status.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study2-position-routing.tar.xz` | `f20c4c978507c887c6a8a3049ef5cfa36f09d350eb913b607167d3e1c9d253e3` | - |
| `studies/position-routing-v2/runs/qualification/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study2-position-routing.tar.xz` | `7c0bdd9a3993133b25c5d940edd2b5b2d9f20573ccee7b0a4d55dfd445ed0cdd` | `studies/position-routing-v2/protocol/CAUSAL-FREEZE.json` |
| `studies/position-routing-v2/runs/replay/summary.json` | redacted (machine details x1); shipped inside `data/saved-outputs/study2-position-routing.tar.xz` | `a334ea2bde2b0fb526b959c928d77eca9becc97eea2df454d327cc7b5a3dd701` | - |

## Files named in freeze records that are not shipped

| Original path | Reason | Original SHA-256 |
|---|---|---|
| `studies/pathway-interactions-v3/analysis/synthetic-tests.log` | log file; logs are not part of the release | `fefd166249373c22bb2d282e0ce97b29629bb483ebf37ff0769f08b9370f8b7d` |
| `studies/position-routing-v2/analysis/synthetic-tests.log` | log file; logs are not part of the release | `fe008aab760eb03ccac7578e2b84665b1b50b0d7f855b17a0c7a8d039435b658` |
