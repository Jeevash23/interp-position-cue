"""Independent checks of text roles and planned fixed-source crosses."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    config = json.loads((ROOT / "03-Data/config/pilot.json").read_text())
    read_rows = lambda name: [json.loads(line) for line in (directory / name).read_text().splitlines()]
    sources = read_rows("sources.jsonl")
    recipients = read_rows("recipients.jsonl")
    plans = read_rows("interventions.jsonl")
    source_by_id, recipient_by_id = {s["id"]: s for s in sources}, {r["id"]: r for r in recipients}
    require(len(source_by_id) == len(sources) and len(recipient_by_id) == len(recipients), "Duplicate prompt IDs")
    require(len({p["id"] for p in plans}) == len(plans), "Duplicate conditions")
    require(len({r["clean_text"] for r in recipients}) == len(recipients), "Duplicate recipient text")
    split_names, split_contexts = defaultdict(set), defaultdict(set)
    for row in sources + recipients:
        io, subject, alternative = row["io"], row["subject"], row["alternative"]
        require(len({io, subject, alternative}) == 3, "Names not distinct")
        clean_words, corrupt_words = re.findall(r"[A-Za-z]+", row["clean_text"]), re.findall(r"[A-Za-z]+", row["corrupted_text"])
        require(len(clean_words) == len(corrupt_words), "Unexpected word insertion/deletion")
        diffs = [(a, b) for a, b in zip(clean_words, corrupt_words) if a != b]
        require(diffs == [(subject, alternative)], "Corruption changes more than repeated subject")
        require(clean_words.count(subject) == 2 and clean_words.count(io) == 1 and alternative not in clean_words, "Wrong clean roles")
        require(corrupt_words.count(subject) == 1 and corrupt_words.count(alternative) == 1, "Wrong corrupt roles")
        initial_names = [w for w in clean_words if w in {io, subject}][:2]
        require(initial_names == ([io, subject] if row["order"] == 0 else [subject, io]), "Mention order mismatch")
        require(row["tokenization_status"] == "not_checked" and row["competence_status"] == "not_measured", "Unperformed qualification claimed")
    blocks = defaultdict(list)
    for row in recipients:
        blocks[row["block_id"]].append(row)
        split_names[row["split"]].update([row["io"], row["subject"], row["alternative"]])
        split_contexts[row["split"]].add((row["place"], row["object"]))
    require(not split_names["development"] & split_names["holdout"], "Name leakage across splits")
    require(not split_contexts["development"] & split_contexts["holdout"], "Context leakage across splits")
    for block in blocks.values():
        require(len(block) == 4 and {(r["family"], r["order"]) for r in block} == {("F10", 0), ("F10", 1), ("F11", 0), ("F11", 1)}, "Incomplete context crossing")
        require(len({(r["io"], r["subject"], r["alternative"]) for r in block}) == 1, "Role identity changes across cells")
    by_recipient, signal_usage, draws = defaultdict(list), defaultdict(set), defaultdict(set)
    for plan in plans:
        require(plan["recipient_id"] in recipient_by_id, "Unknown recipient")
        recipient = recipient_by_id[plan["recipient_id"]]
        require(plan["group_id"] == recipient["group_id"] and plan["split"] == recipient["split"], "Group mismatch")
        require(plan["status"] == "planned_not_started" and plan["signal_hash"] is None, "Unextracted signal claimed")
        by_recipient[recipient["id"]].append(plan)
        if plan["source_id"] is not None:
            require(plan["source_id"] in source_by_id, "Unknown source")
            source = source_by_id[plan["source_id"]]
            require(source["group_id"] == recipient["group_id"], "Source changed name-role group")
            require((source["io"], source["subject"], source["alternative"]) == (recipient["io"], recipient["subject"], recipient["alternative"]), "Source semantic roles differ")
            require((source["place"], source["object"]) != (recipient["place"], recipient["object"]), "Source equals recipient context")
        if plan["condition"] == "fixed_source":
            signal_usage[plan["source_id"]].add((recipient["block_id"], recipient["family"], recipient["order"], plan["strength"]))
        if plan["condition"] == "norm_matched_random":
            draws[plan["random_draw_id"]].add((plan["random_seed"], plan["source_id"]))
    expected_per_recipient = len(config["source_variants"]) * len(config["strengths"]) + 3 + config["random_replicates"]
    for recipient_id, plans_here in by_recipient.items():
        require(len(plans_here) == expected_per_recipient, "Condition count differs")
        conditions = Counter(p["condition"] for p in plans_here)
        require(conditions == Counter(fixed_source=6, untouched_corrupted=1, zero_write=1,
                                     recipient_self=1, norm_matched_random=config["random_replicates"]), "Missing control")
        expected_sources = {s["id"] for s in sources if s["group_id"] == recipient_by_id[recipient_id]["group_id"]}
        require({(p["source_id"], p["strength"]) for p in plans_here if p["condition"] == "fixed_source"} ==
                {(s, strength) for s in expected_sources for strength in config["strengths"]}, "Source/strength cross incomplete")
    require(set(by_recipient) == set(recipient_by_id), "Recipient without conditions")
    require(all(len(values) == 1 for values in draws.values()), "Random direction changes across contexts")
    for source in sources:
        expected = {(r["block_id"], r["family"], r["order"], strength)
                    for r in recipients if r["group_id"] == source["group_id"] for strength in config["strengths"]}
        require(signal_usage[source["id"]] == expected, "Fixed signal not crossed through every matching cell")
    counts = json.loads((directory / "counts.json").read_text())
    require(counts["recipient_prompts"] == len(recipients) and counts["planned_condition_evaluations"] == len(plans), "Incorrect summary count")
    return dict(project_id="AI-08", check="text_and_plan_semantics", status="passed",
                prompt_pairs_checked=len(sources)+len(recipients), fixed_sources=len(sources),
                planned_conditions=len(plans), model_execution=False, tokenizer_loaded=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "03-Data/prepared")
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir), indent=2))
