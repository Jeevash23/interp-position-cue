"""Draw the three study figures of the manuscript from saved outputs.

Sources: the sealed study outputs under studies/ and the post hoc results under analysis/results/ (the same files
that paper/make_numbers.py reads). Outputs: paper/figures/study1.pdf, study2.pdf, study3.pdf and
paper/figures/figure-data.json, which lists every plotted value with its source.
`--check` recomputes the plotted values and exits 1 if they differ from figure-data.json (no drawing).
Run: python -B paper/make_figures.py [--check]

Colours (validated with the data-visualisation palette checker, light mode, all pairs): matched order / benefit
#2a78d6, opposite order / harm #eb6834, queries alone #1baf7a (always with direct value labels), queries plus the
added components #4a3aa7; single-series panels use #2a78d6. Text and error bars use ink colours, never series colours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
S1 = PROJECT / 'studies/fixed-signal-v1'
S2 = PROJECT / 'studies/position-routing-v2'
S3 = PROJECT / 'studies/pathway-interactions-v3'
RES = PROJECT / 'analysis/results'
OUT = PAPER / 'figures'

BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
INK, INK2, MUTED, GRID, NEUTRAL = '#0b0b0b', '#52514e', '#8a8984', '#e6e5e0', '#a3a29d'
BOX = dict(facecolor='white', edgecolor='none', pad=0.5, alpha=0.9)      # keeps value labels legible over lines
STRATA = [('third_name@0.5', 'Third\nname, 0.5'), ('third_name@1', 'Third\nname, 1.0'),
          ('repeat_io@0.5', 'Reversed\nrole, 0.5'), ('repeat_io@1', 'Reversed\nrole, 1.0')]


def read(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).relative_to(PROJECT))


def est(value, lo=None, hi=None, source='', level=None):
    item = dict(value=float(value), source=source)
    if lo is not None:
        item.update(lo=float(lo), hi=float(hi), level=level)
    return item


# ------------------------------------------------------------------------------------------------ data
def data():
    post = read(RES / 'posthoc-intervals.json')
    ph = rel(RES / 'posthoc-intervals.json')
    d = {}
    # Study 1
    f1 = {}
    for bank in ['holdout', 'confirmation']:
        s = read(S1 / 'analysis' / bank / 'summary.json')
        src = rel(S1 / 'analysis' / bank / 'summary.json')
        f1[f'improvement_{bank}'] = est(s['improvement']['mean'], *s['improvement']['ci95'], src, .95)
        f1[f'gain_error_{bank}'] = est(s['gain_normalized_error'], source=src)
        f1[f'directional_error_{bank}'] = est(s['transform_normalized_error'], source=src)
    order = post['study1']['order_split_confirmation']['positive']
    f1['matched'] = est(order['matched']['estimate'], *order['matched']['ci95'], ph, .95)
    f1['opposite'] = est(order['opposite']['estimate'], *order['opposite']['ci95'], ph, .95)
    ctrl = read(S1 / 'analysis/control-descriptions.json')['banks']['confirmation']['behavior_intervals']['positive_repair']
    f1['pooled'] = est(ctrl['mean'], *ctrl['ci95'], rel(S1 / 'analysis/control-descriptions.json'), .95)
    beh = read(S1 / 'analysis/confirmation/summary.json')['behavior']
    for key, condition in [('self', 'recipient_self'), ('random', 'norm_matched_random')]:
        x = post['study1'][condition]
        assert abs(x['estimate'] - beh[condition]['margin_repair']) < 1e-9
        f1[key] = est(x['estimate'], *x['ci95'], ph, .95)
    d['study1'] = f1
    # Study 2
    f2 = {}
    p2 = post['study2']
    for key in ['matched_change', 'opposite_change', 'foreign_matched_change', 'foreign_opposite_change']:
        f2[key] = est(p2[key]['estimate'], *p2[key]['ci95'], ph, .95)
    for mode in ['q', 'k', 'v', 'p', 'z', 'other_heads']:
        x = p2['removal_fraction'][mode]
        f2[f'removed_{mode}'] = est(x['estimate'], *x['ci'], ph, x['level'])
    r = read(S2 / 'analysis/confirmation/summary.json')['ratios']
    for key in ['foreign_retention', 'query_necessity', 'query_sufficiency']:
        f2[key] = est(r[key]['estimate'], *r[key]['interval'], rel(S2 / 'analysis/confirmation/summary.json'), r[key]['interval_level'])
    for key in ['benefit_removed_by_query_clamp', 'harm_removed_by_query_clamp', 'benefit_recreated_by_queries', 'harm_recreated_by_queries']:
        f2[key] = est(p2[key]['estimate'], *p2[key]['ci'], ph, p2[key]['level'])
    d['study2'] = f2
    # Study 3
    f3 = {}
    conf = read(S3 / 'analysis/confirmation/summary.json')
    c3 = rel(S3 / 'analysis/confirmation/summary.json')
    for key, _ in STRATA:
        r = conf['strata'][key]['ratios']
        q = r['clamp_1']
        f3[f'{key}:Q_removal'] = est(1 - q['estimate'], 1 - q['interval'][1], 1 - q['interval'][0], c3, q['interval_level'])
        f3[f'{key}:Q_recreation'] = est(r['restore_1']['estimate'], *r['restore_1']['interval'], c3, r['restore_1']['interval_level'])
        for name, field in [('QAM_removal', 'joint_necessity'), ('QAM_recreation', 'joint_sufficiency'),
                            ('interaction', 'restore_interaction_total')]:
            f3[f'{key}:{name}'] = est(r[field]['estimate'], *r[field]['interval'], c3, r[field]['interval_level'])
    p3 = post['study3']['third_name@1']
    for name, field in [('benefit_removed', 'matched_removed_by_QAM'), ('harm_removed', 'opposite_removed_by_QAM'),
                        ('benefit_recreated', 'matched_recreated_by_QAM'), ('harm_recreated', 'opposite_recreated_by_QAM')]:
        f3[name] = est(p3[field]['estimate'], *p3[field]['ci'], ph, p3[field]['level'])
    d['study3'] = f3
    return d


# --------------------------------------------------------------------------------------------- drawing
def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.edgecolor': INK2,
        'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'legend.frameon': False,
        'svg.hashsalt': 'ai08', 'pdf.compression': 6})
    return plt


def title(ax, letter, text):
    ax.set_title(f'{letter}  {text}', loc='left', fontweight='bold', pad=6)


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def errbar(ax, x, item, scale=1.0, horizontal=False):
    if 'lo' not in item:
        return
    lo, hi, v = item['lo'] * scale, item['hi'] * scale, item['value'] * scale
    if horizontal:
        ax.errorbar(v, x, xerr=[[v - lo], [hi - v]], fmt='none', ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
    else:
        ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)


def study1(plt, d):
    f = d['study1']
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.5, 2.35), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.55))
    rows = [('Pilot holdout\n(128)', 'holdout'), ('Confirmation\n(1,152)', 'confirmation')]
    for i, (label, bank) in enumerate(rows):
        it = f[f'improvement_{bank}']
        a.plot([it['lo'], it['hi']], [i, i], color=BLUE, linewidth=1.6, solid_capstyle='round', zorder=3)
        a.plot([it['value']], [i], 'o', color=BLUE, markersize=5.5, markeredgecolor='white', markeredgewidth=0.8, zorder=4)
        a.text(it['value'], i + 0.2, f"{it['value']:.4f}", ha='center', va='bottom', fontsize=7)
        a.text(0.118, i - 0.27, f"errors {f[f'gain_error_{bank}']['value']:.3f} → {f[f'directional_error_{bank}']['value']:.3f}",
               ha='right', va='top', fontsize=6.5, color=INK2)
    a.axvline(0.05, color=INK2, linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)
    a.text(0.0505, 1.62, 'prespecified 0.05', fontsize=6.5, color=INK2, va='top')
    a.axvline(0, color=INK2, linewidth=0.6)
    a.set_yticks([0, 1], [r[0] for r in rows])
    a.set_ylim(-0.75, 1.7)
    a.set_xlim(-0.005, 0.12)
    a.set_xlabel('Gain error minus directional error')
    grid(a, 'x')
    title(a, 'A', 'Prediction of the response')
    labels = ['Matched order', 'Opposite order', 'Both orders', "Recipient's own signal", 'Random directions']
    items = [f['matched'], f['opposite'], f['pooled'], f['self'], f['random']]
    colors = [BLUE, ORANGE, NEUTRAL, NEUTRAL, NEUTRAL]
    y = list(range(len(labels)))[::-1]
    b.barh(y, [it['value'] for it in items], height=0.56, color=colors, zorder=2)
    for yy, it in zip(y, items):
        errbar(b, yy, it, horizontal=True)
        v = it['value']
        edge = max(v, it.get('hi', v)) if v >= 0 else min(v, it.get('lo', v))
        b.text(edge + (0.08 if v >= 0 else -0.08), yy, f'{v:+.2f}' if abs(v) >= 0.01 else f'{v:+.3f}',
               ha='left' if v >= 0 else 'right', va='center', fontsize=7)
    b.axvline(0, color=INK2, linewidth=0.6)
    b.set_yticks(y, labels)
    b.set_xlim(-2.3, 2.7)
    b.set_xlabel('Change in intended-answer margin (logits)')
    grid(b, 'x')
    title(b, 'B', 'Behaviour at positive strengths')
    fig.savefig(OUT / 'study1.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


def study2(plt, d):
    f = d['study2']
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.5, 2.45), gridspec_kw=dict(width_ratios=[1, 1.25, 1.05], wspace=0.5))
    groups = ['Same names', 'Different names']
    for i, prefix in enumerate(['', 'foreign_']):
        for j, (sign, color) in enumerate([('matched', BLUE), ('opposite', ORANGE)]):
            it = f[f'{prefix}{sign}_change']
            x = i + (j - 0.5) * 0.36
            a.bar(x, it['value'], width=0.34, color=color, zorder=2, label=('Matched' if sign == 'matched' else 'Opposite') if i == 0 else None)
            errbar(a, x, it)
            off = 0.12
            a.text(x, it['hi'] + off if it['value'] > 0 else it['lo'] - off, f"{it['value']:+.2f}", ha='center',
                   va='bottom' if it['value'] > 0 else 'top', fontsize=6.5)
    a.axhline(0, color=INK2, linewidth=0.6)
    a.set_xticks([0, 1], ['Same\nnames', 'Different\nnames'])
    a.set_ylim(-3.3, 4.9)
    a.set_ylabel('Change in margin (logits)')
    a.legend(loc='upper center', ncol=2, handlelength=1.0, borderaxespad=0.1, columnspacing=0.8)
    grid(a)
    title(a, 'A', 'Order effect')
    modes = [('q', 'Queries'), ('k', 'Keys'), ('v', 'Values'), ('p', 'Attention'), ('z', 'Outputs'), ('other_heads', 'Other\nqueries')]
    for i, (mode, label) in enumerate(modes):
        it = f[f'removed_{mode}']
        b.bar(i, 100 * it['value'], width=0.6, color=BLUE, zorder=2)
        errbar(b, i, it, scale=100)
        b.text(i, 100 * max(it['hi'], 0) + 2, f"{100 * it['value']:.1f}", ha='center', va='bottom', fontsize=6.5, bbox=BOX)
    b.axhline(0, color=INK2, linewidth=0.6)
    b.set_xticks(range(len(modes)), [m[1].replace('\n', ' ') for m in modes], rotation=35, ha='right', rotation_mode='anchor')
    b.set_ylim(-8, 62)
    b.set_ylabel('Order contrast removed (%)')
    grid(b)
    title(b, 'B', 'Clamping the five heads')
    ratios = [('foreign_retention', 'Kept with\nother names'), ('query_necessity', 'Removed by\nquery clamp'),
              ('query_sufficiency', 'Recreated\nby queries')]
    y = [2, 1, 0]
    for yy, (key, label) in zip(y, ratios):
        it = f[key]
        c.plot([100 * it['lo'], 100 * it['hi']], [yy, yy], color=BLUE, linewidth=1.6, solid_capstyle='round', zorder=3)
        c.plot([100 * it['value']], [yy], 'o', color=BLUE, markersize=5.5, markeredgecolor='white', markeredgewidth=0.8, zorder=4)
        c.text(100 * it['value'], yy + 0.2, f"{100 * it['value']:.1f}", ha='center', va='bottom', fontsize=6.5, bbox=BOX)
    c.axvline(50, color=INK2, linestyle=(0, (3, 2)), linewidth=0.8)
    c.text(51, -0.62, 'prespecified 50%', fontsize=6.5, color=INK2, va='bottom')
    c.set_yticks(y, [r[1] for r in ratios])
    c.set_xlim(0, 120)
    c.set_ylim(-0.7, 2.6)
    c.set_xlabel('Percent of order contrast')
    grid(c, 'x')
    title(c, 'C', 'Primary ratios')
    fig.savefig(OUT / 'study2.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


def study3(plt, d):
    f, f2 = d['study3'], d['study2']
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.3), gridspec_kw=dict(hspace=0.62, wspace=0.3, top=0.9))
    (a, b), (c, e) = axes
    for ax, kind, letter, text in [(a, 'removal', 'A', 'Removal by clamping'), (b, 'recreation', 'B', 'Recreation by transplanting')]:
        for i, (key, _) in enumerate(STRATA):
            for j, (group, color, label) in enumerate([('Q', AQUA, 'Queries alone (95%)'), ('QAM', VIOLET, 'Q+A+M (99.69%)')]):
                it = f[f'{key}:{group}_{kind}']
                x = i + (j - 0.5) * 0.38
                ax.bar(x, 100 * it['value'], width=0.36, color=color, zorder=2, label=label if i == 0 and ax is a else None)
                errbar(ax, x, it, scale=100)
                ax.text(x, 100 * it['hi'] + 2, f"{100 * it['value']:.0f}", ha='center', va='bottom', fontsize=6.5, bbox=BOX)
        ax.axhline(75, color=INK2, linestyle=(0, (3, 2)), linewidth=0.8, zorder=2.5,
                   label='Prespecified 75%' if ax is a else None)
        ax.set_xticks(range(4), [s[1] for s in STRATA])
        ax.set_ylim(0, 108)
        ax.set_ylabel('Percent of order contrast')
        grid(ax)
        title(ax, letter, text)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.4)
    for i, (key, _) in enumerate(STRATA):
        it = f[f'{key}:interaction']
        c.bar(i, 100 * it['value'], width=0.56, color=BLUE, zorder=2)
        errbar(c, i, it, scale=100)
        c.text(i, 100 * it['lo'] - 1.5, f"{100 * it['value']:.1f}", ha='center', va='top', fontsize=6.5, bbox=BOX)
    c.axhline(0, color=INK2, linewidth=0.6)
    c.set_xticks(range(4), [s[1] for s in STRATA])
    c.set_ylim(-40, 4)
    c.set_ylabel('Joint minus sum (% of contrast)')
    grid(c)
    title(c, 'C', 'Joint transplant minus sum of parts')
    cats = [('Study 2\nremoval', f2['benefit_removed_by_query_clamp'], f2['harm_removed_by_query_clamp']),
            ('Study 2\nrecreation', f2['benefit_recreated_by_queries'], f2['harm_recreated_by_queries']),
            ('Study 3\nremoval', f['benefit_removed'], f['harm_removed']),
            ('Study 3\nrecreation', f['benefit_recreated'], f['harm_recreated'])]
    for i, (label, ben, harm) in enumerate(cats):
        for j, (it, color, name) in enumerate([(ben, BLUE, 'Matched benefit'), (harm, ORANGE, 'Opposite harm')]):
            x = i + (j - 0.5) * 0.38
            e.bar(x, 100 * it['value'], width=0.36, color=color, zorder=2, label=name if i == 0 else None)
            errbar(e, x, it, scale=100)
            e.text(x, 100 * it['hi'] + 2, f"{100 * it['value']:.0f}", ha='center', va='bottom', fontsize=6.5, bbox=BOX)
    e.set_xticks(range(4), [c_[0] for c_ in cats])
    e.set_ylim(0, 100)
    e.set_ylabel('Percent of effect')
    e.legend(loc='upper left', ncol=2, handlelength=1.0, borderaxespad=0.1, columnspacing=1.0)
    grid(e)
    title(e, 'D', 'Benefit and harm, same components')
    fig.savefig(OUT / 'study3.pdf', bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)


def main():
    d = data()
    target = OUT / 'figure-data.json'
    text = json.dumps(d, indent=1, sort_keys=True) + '\n'
    if '--check' in sys.argv:
        if not target.exists() or target.read_text() != text:
            sys.exit('figure data differ from the saved outputs: run make figures')
        print(f'ok  figure data match the saved outputs ({sum(len(v) for v in d.values())} plotted values)')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    plt = setup()
    study1(plt, d)
    study2(plt, d)
    study3(plt, d)
    target.write_text(text)
    print('wrote paper/figures/study1.pdf, study2.pdf, study3.pdf and figure-data.json')


if __name__ == '__main__':
    main()
