"""Prepare text and intervention IDs only; never tokenize or load a model."""
from __future__ import annotations

import hashlib
from support import preparation_main


def render(template, names, context, role, order):
    io, subject = (names[0], names[1]) if role == 0 else (names[1], names[0])
    alternative = names[2]
    a, b = (io, subject) if order == 0 else (subject, io)
    place, obj = context
    fields = dict(place=place, obj=obj, article="an" if obj[0].lower() in "aeiou" else "a",
                  a=a, b=b, s=subject)
    clean = template.format(**fields)
    fields["s"] = alternative
    corrupted = template.format(**fields)
    return dict(clean_text=clean, corrupted_text=corrupted, io=io, subject=subject,
                alternative=alternative, order=order, role=role, place=place, object=obj,
                tokenization_status="not_checked", competence_status="not_measured")


def build(config):
    if config["model_execution_enabled"] or config["tokenization_qualified"] or config["competence_qualified"]:
        raise ValueError("This preparation scaffold accepts only unqualified, unexecuted plans")
    templates = config["templates"]
    if set(templates) != {"F10", "F11"}:
        raise ValueError("The draft requires F10 and F11")
    sources, recipients, interventions = [], [], []
    for split, pools in config["splits"].items():
        for name_index, names in enumerate(pools["name_triples"]):
            for role in (0, 1):
                group = f"{split}:names{name_index}:role{role}"
                source_ids = []
                for variant in config["source_variants"]:
                    source_id = f"{group}:source:{variant['id']}"
                    source_ids.append(source_id)
                    source = dict(id=source_id, group_id=group, split=split,
                        family=variant["family"], variant=variant["id"],
                        **render(templates[variant["family"]], names, config["source_context"],
                                 role, variant["order"]))
                    source["signal_status"] = "not_extracted"
                    sources.append(source)
                for context_index, context in enumerate(pools["contexts"]):
                    block = f"{group}:context{context_index}"
                    for family, template in templates.items():
                        for order in (0, 1):
                            recipient_id = f"{block}:{family}:order{order}"
                            recipients.append(dict(id=recipient_id, block_id=block, group_id=group,
                                split=split, family=family,
                                **render(template, names, context, role, order)))
                            base = dict(recipient_id=recipient_id, block_id=block, group_id=group,
                                        split=split, status="planned_not_started")
                            for source_id in source_ids:
                                for strength in config["strengths"]:
                                    interventions.append(dict(base, id=f"{recipient_id}|{source_id}|{strength}",
                                        condition="fixed_source", source_id=source_id, strength=strength,
                                        signal_hash=None))
                            for control in ("untouched_corrupted", "zero_write", "recipient_self"):
                                interventions.append(dict(base, id=f"{recipient_id}|{control}",
                                    condition=control, source_id=None, strength=None, signal_hash=None))
                            for replicate in range(config["random_replicates"]):
                                draw_id = f"{group}:random:{replicate}"
                                seed_bytes = hashlib.sha256(f"{config['seed']}:{draw_id}".encode()).digest()
                                interventions.append(dict(base, id=f"{recipient_id}|random{replicate}",
                                    condition="norm_matched_random", source_id=source_ids[0], strength=1.0,
                                    random_draw_id=draw_id, random_seed=int.from_bytes(seed_bytes[:4], "big"),
                                    signal_hash=None))
    counts = dict(project_id="AI-08", status="text_preparation_only", source_pairs=len(sources),
                  recipient_prompts=len(recipients),
                  matched_blocks=len({r["block_id"] for r in recipients}),
                  planned_condition_evaluations=len(interventions),
                  extra_calls_not_included=["clean_baselines", "source_caching", "calibration", "mediator_tests"],
                  tokenization_qualified=False, model_runs_completed=0, gpu_used=False,
                  runtime_estimate=None, runner_implemented=False)
    return {"sources.jsonl": sources, "recipients.jsonl": recipients,
            "interventions.jsonl": interventions, "counts.json": counts}


if __name__ == "__main__":
    preparation_main(build)
