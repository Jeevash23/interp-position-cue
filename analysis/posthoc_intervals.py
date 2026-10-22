"""Post hoc crossed-bootstrap intervals for quantities that the frozen analyses report without one.

Every input is a saved study output; nothing is refitted and no model is run. The resampling schemes, seeds and
draw counts are those of each study's own frozen analysis, re-implemented here and checked by first reproducing
saved intervals exactly (the `reproduced` block of the output):

* Study 1 (studies/fixed-signal-v1): ratio of sums over the name-group x context-group grid
  (code/response_models.py, crossed_interval; seed 2026100809; 20,000 draws).
* Study 2 (studies/position-routing-v2): mean of the name-unit x context cell means
  (code/analyze_causal.py, intervals; seed 2026100814; 20,000 draws).
* Study 3 (studies/pathway-interactions-v3): the same scheme with seed 2026100823 (code/analyze.py, crossed).

The `order_contrast_by_wording` block splits the order contrast D of Studies 2 and 3 (strength 1) by sentence wording,
with the same schemes, to show where the change in D between data sets comes from.

All new intervals are pointwise 95% percentile intervals and are descriptive. Writes
analysis/results/posthoc-intervals.json. Run from anywhere: python -B analysis/posthoc_intervals.py
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
S1 = PROJECT / 'studies/fixed-signal-v1'
S2 = PROJECT / 'studies/position-routing-v2'
S3 = PROJECT / 'studies/pathway-interactions-v3'
OUT = PROJECT / 'analysis/results/posthoc-intervals.json'
DRAWS = 20000
INPUTS = []


def read(path):
    INPUTS.append(path)
    return json.loads(Path(path).read_text())


def load(path):
    INPUTS.append(path)
    return np.load(path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def close(a, b, tol=1e-12):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


# --------------------------------------------------------------------------------------- resampling schemes
def ratio_of_sums(numerator, denominator, names, contexts, seed=2026100809, draws=DRAWS):
    """Study 1's scheme: resample name groups and context groups; statistic = sum(num) / sum(den)."""
    ns, cs = sorted(set(names)), sorted(set(contexts))
    ng = np.zeros((len(ns), len(cs)))
    dg = ng.copy()
    for i, n in enumerate(ns):
        for j, c in enumerate(cs):
            mask = (names == n) & (contexts == c)
            ng[i, j] = np.sum(numerator[mask])
            dg[i, j] = np.sum(denominator[mask])
    rng = np.random.default_rng(seed)
    a = rng.integers(len(ns), size=(draws, len(ns)))
    b = rng.integers(len(cs), size=(draws, len(cs)))
    boot = ng[a[:, :, None], b[:, None, :]].sum((1, 2)) / np.maximum(dg[a[:, :, None], b[:, None, :]].sum((1, 2)), 1e-30)
    return dict(estimate=float(np.sum(numerator) / np.sum(denominator)), ci95=np.quantile(boot, [.025, .975]).tolist())


def cell_mean_boot(series, names, contexts, seed, draws=DRAWS):
    """Studies 2-3: grid of cell means; each draw averages the resampled grid. Returns estimates and draws."""
    ns, cs = sorted(set(names.tolist())), sorted(set(contexts.tolist()))
    ni = np.array([ns.index(x) for x in names])
    ci = np.array([cs.index(x) for x in contexts])
    counts = np.zeros((len(ns), len(cs)))
    np.add.at(counts, (ni, ci), 1)
    assert np.all(counts > 0)
    grids = {}
    for k, v in series.items():
        g = np.zeros((len(ns), len(cs)))
        np.add.at(g, (ni, ci), v)
        grids[k] = g / counts
    rng = np.random.default_rng(seed)
    a = rng.integers(len(ns), size=(draws, len(ns)))
    b = rng.integers(len(cs), size=(draws, len(cs)))
    boot = {k: g[a[:, :, None], b[:, None, :]].mean((1, 2)) for k, g in grids.items()}
    est = {k: float(g.mean()) for k, g in grids.items()}
    return est, boot


def ratio(est, boot, num, den, level=.95):
    values = boot[num] / boot[den]
    alpha = 1 - level
    return dict(estimate=est[num] / est[den], ci=np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist(),
                level=level, denominators_at_or_below_025=int(np.sum(np.abs(boot[den]) <= .25)))


def mean_ci(est, boot, key):
    return dict(estimate=est[key], ci95=np.quantile(boot[key], [.025, .975]).tolist())


# ------------------------------------------------------------------------------------------------- Study 1
def study1():
    out, checks = {}, {}
    for bank in ['holdout', 'confirmation']:
        folder = S1 / 'analysis' / bank
        pairs = read(folder / 'pairs.json')
        err = load(folder / 'prediction-errors.npz')
        names = np.array([p['name'] for p in pairs])
        contexts = np.array([p['context'] for p in pairs])
        gain, transform, energy = err['gain'], err['transform'], err['energy']
        saved = read(folder / 'summary.json')
        absolute = ratio_of_sums(gain - transform, energy, names, contexts)
        checks[f'study1_{bank}_absolute_improvement'] = close(absolute['estimate'], saved['improvement']['mean']) and \
            close(absolute['ci95'], saved['improvement']['ci95'])
        out[bank] = dict(absolute_improvement=absolute,
                         relative_improvement=ratio_of_sums(gain - transform, gain, names, contexts),
                         gain_error=float(gain.sum() / energy.sum()), transform_error=float(transform.sum() / energy.sum()))
    # Order split of the positive-strength fixed-source edits (confirmation): one value per recipient.
    behaviour = read(S1 / 'analysis/confirmation/behavior.json')
    per = defaultdict(lambda: defaultdict(list))
    meta = {}
    for r in behaviour:
        if r['condition'] == 'fixed_source':
            matched = r['cell'].endswith('order0') == (r['source_variant'] == 'canonical')
            key = ('matched' if matched else 'opposite')
            per[r['row_id']][(key, r['strength'])].append(r['margin_repair'])
            meta[r['row_id']] = (r['name'], r['context'])
    rows = sorted(per)
    names = np.array([meta[r][0] for r in rows])
    contexts = np.array([meta[r][1] for r in rows])
    one = np.ones(len(rows))
    order = {}
    for label, strengths in [('positive', [.5, 1.]), ('plus_half', [.5]), ('plus_one', [1.]), ('minus_half', [-.5])]:
        m = np.array([np.mean([v for s in strengths for v in per[r][('matched', s)]]) for r in rows])
        o = np.array([np.mean([v for s in strengths for v in per[r][('opposite', s)]]) for r in rows])
        hm = np.array([np.mean([v < 0 for s in strengths for v in per[r][('matched', s)]]) for r in rows])
        ho = np.array([np.mean([v < 0 for s in strengths for v in per[r][('opposite', s)]]) for r in rows])
        order[label] = dict(matched=ratio_of_sums(m, one, names, contexts), opposite=ratio_of_sums(o, one, names, contexts),
                            contrast=ratio_of_sums(m - o, one, names, contexts),
                            harmful_fraction_matched=ratio_of_sums(hm, one, names, contexts),
                            harmful_fraction_opposite=ratio_of_sums(ho, one, names, contexts))
    controls = read(S1 / 'analysis/control-descriptions.json')['banks']['confirmation']['positive_mention_order']
    checks['study1_order_means'] = close(order['positive']['matched']['estimate'], controls['matching_mention_order']['margin_repair'], 1e-9) and \
        close(order['positive']['opposite']['estimate'], controls['opposite_mention_order']['margin_repair'], 1e-9)
    out['order_split_confirmation'] = order
    # Reference conditions of Figure 2B: the recipient's own signal and the four norm-matched random directions.
    refs = defaultdict(lambda: defaultdict(list))
    for r in behaviour:
        if r['condition'] in ('recipient_self', 'norm_matched_random'):
            refs[r['row_id']][r['condition']].append(r['margin_repair'])
    summary = read(S1 / 'analysis/confirmation/summary.json')['behavior']
    for condition in ['recipient_self', 'norm_matched_random']:
        values = np.array([np.mean(refs[r][condition]) for r in rows])
        out[condition] = ratio_of_sums(values, one, names, contexts)
        checks[f'study1_{condition}_mean'] = close(out[condition]['estimate'], summary[condition]['margin_repair'], 1e-9)
    out['recipients'] = len(rows)
    return out, checks


# ------------------------------------------------------------------------------------------------- Study 2
def study2():
    summary = read(S2 / 'analysis/confirmation/summary.json')
    labels = summary['labels']
    col = {label: i for i, label in enumerate(labels)}
    rows = read(S2 / 'runs/confirmation/recipients.json')
    z = load(S2 / 'analysis/confirmation/paired-scores.npz')
    margins = z['margins']
    names = np.array([r['name_index'] for r in rows])
    contexts = np.array([r['context_index'] for r in rows])

    def change(label):
        return margins[:, col[label]] - margins[:, col['zero']]

    def contrast(kind, suffix=''):
        return margins[:, col[f'{kind}_matched{suffix}']] - margins[:, col[f'{kind}_opposite{suffix}']]

    series = {'D': contrast('same'), 'D_foreign': contrast('foreign')}
    for mode in ['q', 'k', 'v', 'p', 'z']:
        series[f'clamp_{mode}'] = contrast('same', f':clamp_{mode}')
        series[f'foreign_clamp_{mode}'] = contrast('foreign', f':clamp_{mode}')
    series['clamp_other_heads'] = (contrast('same', ':control0_q') + contrast('same', ':control1_q')) / 2
    series['query_only'] = contrast('same', ':query_only')
    series['foreign_query_only'] = contrast('foreign', ':query_only')
    for mode in ['q', 'k', 'v', 'p', 'z']:
        series[f'removed_{mode}'] = series['D'] - series[f'clamp_{mode}']
        series[f'foreign_removed_{mode}'] = series['D_foreign'] - series[f'foreign_clamp_{mode}']
    series['removed_other_heads'] = series['D'] - series['clamp_other_heads']
    for sign in ['matched', 'opposite']:
        series[f'foreign_{sign}_change'] = change(f'foreign_{sign}')
        series[f'{sign}_change'] = change(f'same_{sign}')
        series[f'{sign}_change_clamp_q'] = change(f'same_{sign}:clamp_q')
        series[f'{sign}_removed_q'] = series[f'{sign}_change'] - series[f'{sign}_change_clamp_q']
        series[f'{sign}_query_only'] = change(f'same_{sign}:query_only')
    est, boot = cell_mean_boot(series, names, contexts, seed=2026100814)
    checks = {'study2_D': close(est['D'], summary['absolute_contrasts']['same']['mean'], 1e-9) and
              close(np.quantile(boot['D'], [.025, .975]), summary['absolute_contrasts']['same']['ci95'], 1e-9),
              'study2_query_necessity': close(ratio(est, boot, 'removed_q', 'D', 1 - .05 / 3)['ci'],
                                              summary['ratios']['query_necessity']['interval'], 1e-9)}
    out = dict(removal_fraction={m: ratio(est, boot, f'removed_{m}', 'D') for m in ['q', 'k', 'v', 'p', 'z', 'other_heads']},
               foreign_removal_fraction={m: ratio(est, boot, f'foreign_removed_{m}', 'D_foreign') for m in ['q', 'k', 'v', 'p', 'z']},
               foreign_query_only_fraction=ratio(est, boot, 'foreign_query_only', 'D_foreign'),
               benefit_removed_by_query_clamp=ratio(est, boot, 'matched_removed_q', 'matched_change'),
               harm_removed_by_query_clamp=ratio(est, boot, 'opposite_removed_q', 'opposite_change'),
               benefit_recreated_by_queries=ratio(est, boot, 'matched_query_only', 'matched_change'),
               harm_recreated_by_queries=ratio(est, boot, 'opposite_query_only', 'opposite_change'),
               matched_change=mean_ci(est, boot, 'matched_change'), opposite_change=mean_ci(est, boot, 'opposite_change'),
               foreign_matched_change=mean_ci(est, boot, 'foreign_matched_change'),
               foreign_opposite_change=mean_ci(est, boot, 'foreign_opposite_change'))
    return out, checks


# ------------------------------------------------------------------------------------------------- Study 3
def study3():
    summary = read(S3 / 'analysis/confirmation/summary.json')
    plan = read(S3 / 'runs/confirmation/conditions.json')
    col = {p['id']: i for i, p in enumerate(plan)}
    rows = read(S3 / 'runs/confirmation/recipients.json')
    margins = load(S3 / 'analysis/confirmation/paired-scores.npz')['margins']
    out, checks = {}, {}
    for corruption in ['third_name', 'repeat_io']:
        take = np.array([i for i, r in enumerate(rows) if r['corruption'] == corruption])
        mm = margins[take]
        names = np.array([rows[i]['name_index'] for i in take])
        contexts = np.array([rows[i]['context_index'] for i in take])
        for dose in [.5, 1.]:
            key = f'{corruption}@{dose:g}'
            series = {}
            for sign in ['matched', 'opposite']:
                base = mm[:, col['zero']]
                for kind, mask in [('clamp', 0), ('clamp', 1), ('clamp', 7), ('restore', 1), ('restore', 7)]:
                    series[f'{sign}_{kind}{mask}'] = mm[:, col[f'{kind}:{mask}:{sign}@{dose:g}']] - base
                series[f'{sign}_removed7'] = series[f'{sign}_clamp0'] - series[f'{sign}_clamp7']
                series[f'{sign}_removed1'] = series[f'{sign}_clamp0'] - series[f'{sign}_clamp1']
            series['D'] = series['matched_clamp0'] - series['opposite_clamp0']
            est, boot = cell_mean_boot(series, names, contexts, seed=2026100823)
            saved = summary['strata'][key]['absolute']['clamp_0']
            checks[f'study3_{key}_D'] = close(est['D'], saved['mean'], 1e-9) and \
                close(np.quantile(boot['D'], [.025, .975]), saved['ci95'], 1e-9)
            out[key] = {f'{sign}_{what}': ratio(est, boot, f'{sign}_{num}', f'{sign}_clamp0')
                        for sign in ['matched', 'opposite']
                        for what, num in [('removed_by_QAM', 'removed7'), ('removed_by_Q', 'removed1'),
                                          ('recreated_by_QAM', 'restore7'), ('recreated_by_Q', 'restore1')]}
            out[key].update(matched_change=mean_ci(est, boot, 'matched_clamp0'), opposite_change=mean_ci(est, boot, 'opposite_clamp0'))
    return out, checks


# ------------------------------------------------------------------------------- order contrast by wording
def per_wording(contrast, rows, seed):
    """Order contrast D (mean of name-unit x context cell means) within each wording, with 95% intervals."""
    families = np.array([r['family'] for r in rows])
    names = np.array([r['name_index'] for r in rows])
    contexts = np.array([r['context_index'] for r in rows])
    out = {}
    for family in sorted(set(families.tolist())) + ['all']:
        sub = np.ones(len(rows), bool) if family == 'all' else families == family
        est, boot = cell_mean_boot({'D': contrast[sub]}, names[sub], contexts[sub], seed)
        out[family] = dict(mean_ci(est, boot, 'D'), recipients=int(sub.sum()))
    return out


def wording():
    """Post hoc: the order contrast D separately for each wording (Studies 2 and 3, strength 1)."""
    out = {}
    for bank in ['development', 'confirmation']:
        labels = read(S2 / 'analysis' / bank / 'summary.json')['labels']
        col = {label: i for i, label in enumerate(labels)}
        rows = read(S2 / 'runs' / bank / 'recipients.json')
        margins = load(S2 / 'analysis' / bank / 'paired-scores.npz')['margins']
        contrast = margins[:, col['same_matched']] - margins[:, col['same_opposite']]
        out[f'study2_{bank}'] = per_wording(contrast, rows, seed=2026100814)
    for bank in ['development', 'confirmation']:
        plan = read(S3 / 'runs' / bank / 'conditions.json')
        col = {p['id']: i for i, p in enumerate(plan)}
        rows = read(S3 / 'runs' / bank / 'recipients.json')
        margins = load(S3 / 'analysis' / bank / 'paired-scores.npz')['margins']
        for corruption in ['third_name', 'repeat_io']:
            take = [i for i, r in enumerate(rows) if r['corruption'] == corruption]
            contrast = margins[take, col['clamp:0:matched@1']] - margins[take, col['clamp:0:opposite@1']]
            out[f'study3_{bank}_{corruption}@1'] = per_wording(contrast, [rows[i] for i in take], seed=2026100823)
    checks = {'wording_study2_confirmation_all': close(out['study2_confirmation']['all']['estimate'],
                                                       read(S2 / 'analysis/confirmation/summary.json')['absolute_contrasts']['same']['mean'], 1e-9),
              'wording_study3_confirmation_all': close(out['study3_confirmation_third_name@1']['all']['estimate'],
                                                       read(S3 / 'analysis/confirmation/summary.json')['strata']['third_name@1']['absolute']['clamp_0']['mean'], 1e-9)}
    return out, checks


def main():
    s1, c1 = study1()
    s2, c2 = study2()
    s3, c3 = study3()
    w, c4 = wording()
    checks = {**c1, **c2, **c3, **c4}
    if not all(checks.values()):
        raise SystemExit(f'reproduction of saved intervals failed: {checks}')
    result = dict(
        description='Post hoc 95% crossed-bootstrap intervals computed from saved outputs with each study\'s own '
                    'resampling scheme and seed; descriptive, not part of any frozen protocol.',
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        reproduced=checks, study1=s1, study2=s2, study3=s3, order_contrast_by_wording=w,
        inputs={str(Path(p).relative_to(PROJECT)): sha(p) for p in sorted(set(map(str, INPUTS)))})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(OUT.read_text()) if OUT.exists() else None
    if previous is not None:
        previous.pop('created_utc', None)
        current = dict(result)
        current.pop('created_utc')
        if previous == json.loads(json.dumps(current)):
            print('posthoc-intervals.json unchanged')
            return
    OUT.write_text(json.dumps(result, indent=1) + '\n')
    print('wrote', OUT.relative_to(PROJECT), '; reproduced saved intervals:', all(checks.values()))


if __name__ == '__main__':
    main()
