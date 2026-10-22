"""Post hoc: do the two added heads (15.4, 12.3) feed the five known queries? (Study 3 saved traces; CPU only.)

Study 3 retained complete final-position traces (queries, head outputs, MLP outputs of blocks 7-23) for every
condition on a sample fixed before measurement: 16 recipients per factorial bank (first/last name unit x first/last
context x two wordings x two corruptions; role 0, order 0). For each corruption and correction strength, this script
takes the order-dependent change of a variable,  delta = value(matched source) - value(opposite source),  and asks how
much of the change seen under the full correction survives when another group is held at its baseline value:

    surviving fraction = sum_r <delta_r(clamped), delta_r(full)> / sum_r ||delta_r(full)||^2   (r = recipients)

and how much a transplant of that group alone (no source correction) recreates:

    recreated fraction = sum_r <delta_r(transplant), delta_r(full)> / sum_r ||delta_r(full)||^2.

Variables: the five query heads' final-position queries (Q); the two added heads' outputs (A). Groups: A clamped
(condition clamp:2), M clamped (clamp:4), Q clamped (clamp:1); transplants restore:2 and restore:4. Built-in check:
head 12.3 lies in block 12, before every query head, so holding the queries fixed must leave its change exactly
unchanged. The intervention semantics follow code/path_engine.py: a clamped site takes the recipient's own
uncorrected value; everything downstream is recomputed. Descriptive only: 8 recipients per stratum, no interval.

Writes analysis/results/query-feed.json. Run: python -B analysis/posthoc_query_feed.py [--bank development]
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
S3 = PROJECT / 'studies/pathway-interactions-v3'
LAYERS = list(range(7, 24))                                   # capture layers of the saved traces
QUERY_HEADS = [(14, 14), (15, 14), (16, 15), (17, 4), (19, 15)]
ADDED_HEADS = [(15, 4), (12, 3)]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def pick(trace, heads):
    """trace: (conditions, layers, heads, 64) -> (conditions, len(heads), 64)."""
    return np.stack([trace[:, LAYERS.index(layer), head] for layer, head in heads], axis=1).astype(np.float64)


def fraction(test, full):
    """Pooled projection of the tested change onto the full change; test/full: (recipients, ..., 64)."""
    num = float(np.sum(test * full))
    den = float(np.sum(full * full))
    return num / den


def main(bank):
    run = S3 / 'runs' / bank
    plan = json.loads((run / 'conditions.json').read_text())
    rows = json.loads((run / 'recipients.json').read_text())
    col = {p['id']: i for i, p in enumerate(plan)}
    files = sorted(run.glob('row-*.npz'))
    traced = []
    for f in files:
        with np.load(f) as z:
            if 'trace_q' in z.files:
                traced.append(f)
    out = dict(description=__doc__.strip().splitlines()[0], bank=bank, traced_recipients=len(traced), strata={},
               inputs={})
    data = []
    for f in traced:
        i = int(f.stem.split('-')[1])
        with np.load(f) as z:
            data.append((rows[i], pick(z['trace_q'], QUERY_HEADS), pick(z['trace_z'], ADDED_HEADS)))
        out['inputs'][str(f.relative_to(PROJECT))] = sha(f)
    check_ok = True
    for corruption in ['third_name', 'repeat_io']:
        sub = [d for d in data if d[0]['corruption'] == corruption]
        for dose in [.5, 1.]:
            def delta(kind, mask, which):
                arrays = []
                for _, q, a in sub:
                    x = q if which == 'q' else a
                    arrays.append(x[col[f'{kind}:{mask}:matched@{dose:g}']] - x[col[f'{kind}:{mask}:opposite@{dose:g}']])
                return np.stack(arrays)
            q_full, a_full = delta('clamp', 0, 'q'), delta('clamp', 0, 'a')
            res = dict(recipients=len(sub))
            res['queries_surviving_when_A_held'] = fraction(delta('clamp', 2, 'q'), q_full)
            res['queries_surviving_when_M_held'] = fraction(delta('clamp', 4, 'q'), q_full)
            res['queries_surviving_when_A_and_M_held'] = fraction(delta('clamp', 6, 'q'), q_full)
            res['queries_recreated_by_A_transplant'] = fraction(delta('restore', 2, 'q'), q_full)
            res['queries_recreated_by_M_transplant'] = fraction(delta('restore', 4, 'q'), q_full)
            res['queries_surviving_when_A_held_by_head'] = {
                f'{l}.{h}': fraction(delta('clamp', 2, 'q')[:, k], q_full[:, k]) for k, (l, h) in enumerate(QUERY_HEADS)}
            a_when_q = delta('clamp', 1, 'a')
            res['added_head_output_surviving_when_Q_held_by_head'] = {
                f'{l}.{h}': fraction(a_when_q[:, k], a_full[:, k]) for k, (l, h) in enumerate(ADDED_HEADS)}
            # Structural check: 12.3 (block 12) precedes all query heads, so clamping queries cannot change it.
            k123 = ADDED_HEADS.index((12, 3))
            exact = bool(np.array_equal(a_when_q[:, k123], a_full[:, k123]))
            res['check_12_3_unchanged_when_Q_held'] = exact
            check_ok &= exact
            out['strata'][f'{corruption}@{dose:g}'] = res
    out['checks_passed'] = check_ok
    out['created_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    target = PROJECT / 'analysis/results' / ('query-feed.json' if bank == 'confirmation' else f'query-feed-{bank}.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(target.read_text()) if target.exists() else None
    if previous is not None:
        previous.pop('created_utc', None)
        current = json.loads(json.dumps({k: v for k, v in out.items() if k != 'created_utc'}))
        if previous == current:
            print(target.name, 'unchanged')
            return
    target.write_text(json.dumps(out, indent=1) + '\n')
    print('wrote', target.relative_to(PROJECT), '; structural check passed:', check_ok)
    if not check_ok:
        sys.exit(1)


if __name__ == '__main__':
    main(sys.argv[sys.argv.index('--bank') + 1] if '--bank' in sys.argv else 'confirmation')
