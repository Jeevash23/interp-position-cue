# Model checkpoint

No weights are included. All three studies use GPT-2 Medium, read from `models/gpt2-medium` or from the folder
named by `MODELS_DIR`:

| Subfolder | Hugging Face repository | Revision | Size |
|---|---|---|---|
| `gpt2-medium` | `openai-community/gpt2-medium` | `6dcaa7a952f72f9298047fd5137cd6e4f05f41da` | 1.52 GB |

`models.json` lists every file with the SHA-256 recorded before the runs; each study's
`provenance/MODEL-ASSETS.json` records the same eight asset files, and the study code checks them again before it
loads the model.

```sh
python tools/get_models.py gpt2-medium                 # download and check (make models)
python tools/get_models.py --check                     # check what is here, without network access
python tools/get_models.py --link-from /path/to/models # use a checkpoint you already have, through a link
```

GPT-2 Medium is distributed under its own licence (MIT).
