"""Write every number in the manuscript as a LaTeX macro, and the generated tables, from saved outputs.

Sources: the sealed study outputs under studies/ (never modified) and the post hoc results under analysis/results/.
Outputs: paper/generated/numbers.tex and paper/generated/table-*.tex.
`--check` recomputes everything and exits 1 if any generated file differs from what is on disk.
Run: python -B paper/make_numbers.py [--check]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent
PROJECT = PAPER.parent
S1 = PROJECT / 'studies/fixed-signal-v1'
S2 = PROJECT / 'studies/position-routing-v2'
S3 = PROJECT / 'studies/pathway-interactions-v3'
RES = PROJECT / 'analysis/results'
GEN = PAPER / 'generated'

MACROS: dict[str, str] = {}
SOURCES: dict[str, str] = {}


def read(path):
    return json.loads(Path(path).read_text())


def put(name, value, source):
    assert re.fullmatch(r'[A-Za-z]+', name), name
    assert name not in MACROS, f'duplicate macro {name}'
    MACROS[name] = value
    SOURCES[name] = source


def fmt(x, d):
    s = f'{x:,.{d}f}'
    return s.replace('-', '\\textminus ') if s.startswith('-') else s


def pct(x, d=1):
    return fmt(100 * x, d)


def count(n):
    return f'{int(n):,}'.replace(',', '{,}')


def number(name, x, d, source):
    put(name, fmt(x, d), source)


def percent(name, x, source, d=1):
    put(name, pct(x, d), source)


def interval(name, est, lo, hi, d, source, scale=1.0):
    """\\Name = estimate; \\NameLo, \\NameHi = bounds; \\NameCI = '[lo, hi]'."""
    put(name, fmt(scale * est, d), source)
    put(name + 'Lo', fmt(scale * lo, d), source)
    put(name + 'Hi', fmt(scale * hi, d), source)
    put(name + 'CI', f'[{fmt(scale * lo, d)}, {fmt(scale * hi, d)}]', source)


def pinterval(name, est, ci, source, d=1):
    interval(name, est, ci[0], ci[1], d, source, scale=100.0)


# ============================================================================================== Study 1
def study1():
    src = 'studies/fixed-signal-v1/analysis/'
    conf = read(S1 / 'analysis/confirmation/summary.json')
    hold = read(S1 / 'analysis/holdout/summary.json')
    summ = read(S1 / 'analysis/STUDY-SUMMARY.json')
    ctrl = read(S1 / 'analysis/control-descriptions.json')['banks']
    sel = read(S1 / 'analysis/development/selection.json')
    post = read(RES / 'posthoc-intervals.json')['study1']
    c = src + 'confirmation/summary.json'
    number('OneGainErr', conf['gain_normalized_error'], 3, c)
    number('OneDirErr', conf['transform_normalized_error'], 3, c)
    interval('OneImp', conf['improvement']['mean'], *conf['improvement']['ci95'], 4, c)
    pinterval('OneRel', post['confirmation']['relative_improvement']['estimate'], post['confirmation']['relative_improvement']['ci95'],
              'analysis/results/posthoc-intervals.json')
    interval('OnePerp', conf['perpendicular_improvement']['mean'], *conf['perpendicular_improvement']['ci95'], 4, c)
    put('OneNamesPos', str(sum(v > 0 for v in conf['improvement']['name_effects'])), c)
    put('OneCtxPos', str(sum(v > 0 for v in conf['improvement']['context_effects'])), c)
    put('OneGroups', str(conf['improvement']['names']), c)
    percent('OneOracle', conf['oracle_scalar_fractional_error'], c)
    number('OneCosine', conf['direction_coverage']['cosine_mean'], 2, c)
    put('OneRecipients', count(conf['recipients']), c)
    put('OnePairs', count(conf['prediction_pairs']), c)
    put('OneRank', str(conf['model_rank']), c)
    put('OneRidge', f"{conf['ridge']:g}", c)
    put('OneMaxRank', str(max(int(k.strip('()').split(',')[0]) for k in sel['candidates'])), src + 'development/selection.json')
    h = src + 'holdout/summary.json'
    number('OneHoldGainErr', hold['gain_normalized_error'], 3, h)
    number('OneHoldDirErr', hold['transform_normalized_error'], 3, h)
    interval('OneHoldImp', hold['improvement']['mean'], *hold['improvement']['ci95'], 4, h)
    pinterval('OneHoldRel', post['holdout']['relative_improvement']['estimate'], post['holdout']['relative_improvement']['ci95'],
              'analysis/results/posthoc-intervals.json')
    cell = hold['competence']['recipients']['F11/order1']
    put('OneHoldFailCorrect', str(round(cell['top1'] * cell['n'])), h)
    put('OneHoldFailN', str(cell['n']), h)
    percent('OneHoldFailPct', cell['top1'], h)
    percent('OneCvRel', summ['development_cross_validation_relative_improvement'], src + 'STUDY-SUMMARY.json')
    energy = summ['banks']['confirmation']['primary_energy_by_strength']
    percent('OneEnergyOne', energy['1.0'], src + 'STUDY-SUMMARY.json')
    percent('OneEnergyHalf', energy['0.5'], src + 'STUDY-SUMMARY.json')
    percent('OneEnergyNeg', energy['-0.5'], src + 'STUDY-SUMMARY.json', d=1)
    sl = ctrl['confirmation']['prediction_slices']
    cd = src + 'control-descriptions.json'
    number('OneSliceNeg', sl['strength:-0.5']['improvement'], 3, cd)
    number('OneSliceHalf', sl['strength:0.5']['improvement'], 3, cd)
    number('OneSliceOne', sl['strength:1.0']['improvement'], 3, cd)
    number('OneDoseLin', ctrl['confirmation']['dose_linearity_diagnostic']['one_vs_twice_half']['normalized_error'], 2, cd)
    beh = conf['behavior']
    order = ctrl['confirmation']['positive_mention_order']
    percent('OneUntouchedTop', beh['untouched_corrupted']['correct'], c)
    percent('OneMatchedTop', order['matching_mention_order']['correct'], cd)
    percent('OneOppositeTop', order['opposite_mention_order']['correct'], cd)
    percent('OnePooledTop', ctrl['confirmation']['positive_strength']['correct'], cd)
    number('OneSelf', beh['recipient_self']['margin_repair'], 2, c)
    put('OneSelfCI', f"[{fmt(post['recipient_self']['ci95'][0], 2)}, {fmt(post['recipient_self']['ci95'][1], 2)}]",
        'analysis/results/posthoc-intervals.json')
    put('OneRandomCI', f"[{fmt(post['norm_matched_random']['ci95'][0], 3)}, {fmt(post['norm_matched_random']['ci95'][1], 3)}]",
        'analysis/results/posthoc-intervals.json')
    percent('OneSelfTop', beh['recipient_self']['correct'], c)
    number('OneRandom', beh['norm_matched_random']['margin_repair'], 3, c)
    percent('OneRandomTop', beh['norm_matched_random']['correct'], c)
    bi = ctrl['confirmation']['behavior_intervals']
    interval('OnePooled', bi['positive_repair']['mean'], *bi['positive_repair']['ci95'], 3, cd)
    interval('OnePooledVsRandom', bi['positive_minus_random']['mean'], *bi['positive_minus_random']['ci95'], 3, cd)
    number('OneKLPooled', ctrl['confirmation']['positive_strength']['kl_from_corrupted'], 2, cd)
    number('OneKLRandom', beh['norm_matched_random']['kl_from_corrupted'], 4, c)
    number('OneKLSelf', beh['recipient_self']['kl_from_corrupted'], 2, c)
    o = post['order_split_confirmation']
    ph = 'analysis/results/posthoc-intervals.json'
    interval('OneMatched', o['positive']['matched']['estimate'], *o['positive']['matched']['ci95'], 2, ph)
    interval('OneOpposite', o['positive']['opposite']['estimate'], *o['positive']['opposite']['ci95'], 2, ph)
    lo, hi = o['positive']['opposite']['ci95']
    interval('OneOppositeMag', -o['positive']['opposite']['estimate'], -hi, -lo, 2, ph)
    interval('OneContrast', o['positive']['contrast']['estimate'], *o['positive']['contrast']['ci95'], 2, ph)
    interval('OneContrastOne', o['plus_one']['contrast']['estimate'], *o['plus_one']['contrast']['ci95'], 2, ph)
    interval('OneMatchedOne', o['plus_one']['matched']['estimate'], *o['plus_one']['matched']['ci95'], 2, ph)
    interval('OneOppositeOne', o['plus_one']['opposite']['estimate'], *o['plus_one']['opposite']['ci95'], 2, ph)
    pinterval('OneHarmOpp', o['positive']['harmful_fraction_opposite']['estimate'], o['positive']['harmful_fraction_opposite']['ci95'], ph)
    pinterval('OneHarmMatched', o['positive']['harmful_fraction_matched']['estimate'], o['positive']['harmful_fraction_matched']['ci95'], ph, d=2)
    percent('OneHarmOppSaved', order['opposite_mention_order']['harmful_margin_fraction'], cd, d=2)
    comp = conf['competence']['recipients']
    put('OneCompMin', pct(min(v['top1'] for v in comp.values())), c)
    gains = read(S1 / 'analysis/development/model.json')['gains']          # 'cell|source|strength' -> fitted gain
    md = src + 'development/model.json'
    number('OneGainMin', min(gains.values()), 2, md)
    number('OneGainMax', max(gains.values()), 2, md)
    flip = [v for k, v in gains.items() if k.startswith('F10/order1|')]   # same wording, reversed recipient order
    number('OneGainFlipMin', min(flip), 2, md)
    number('OneGainFlipMax', max(flip), 2, md)
    put('OneTotalRecipients', count(summ['total_recipient_cases']), src + 'STUDY-SUMMARY.json')
    put('OneRecords', count(summ['total_condition_records']), src + 'STUDY-SUMMARY.json')
    put('OneSourcePairs', count(summ['total_source_pairs']), src + 'STUDY-SUMMARY.json')
    put('OneExamples', count(summ['model_examples']['total']), src + 'STUDY-SUMMARY.json')
    put('OneHoldRecipients', count(hold['recipients']), h)


# ============================================================================================== Study 2
def study2():
    src = 'studies/position-routing-v2/analysis/'
    conf = read(S2 / 'analysis/confirmation/summary.json')
    dev = read(S2 / 'analysis/development/summary.json')
    summ = read(S2 / 'analysis/STUDY-SUMMARY.json')
    post = read(RES / 'posthoc-intervals.json')['study2']
    c = src + 'confirmation/summary.json'
    ph = 'analysis/results/posthoc-intervals.json'
    ac = conf['absolute_contrasts']
    interval('TwoD', ac['same']['mean'], *ac['same']['ci95'], 2, c)
    interval('TwoDForeign', ac['foreign']['mean'], *ac['foreign']['ci95'], 2, c)
    number('TwoDRaw', ac['raw_same']['mean'], 2, c)
    number('TwoDRawForeign', ac['raw_foreign']['mean'], 2, c)
    r = conf['ratios']
    for name, key in [('TwoRetain', 'foreign_retention'), ('TwoNec', 'query_necessity'), ('TwoSuf', 'query_sufficiency'),
                      ('TwoSpecRandom', 'specificity_vs_random_heads'), ('TwoSpecValues', 'specificity_vs_values'),
                      ('TwoSpecRotation', 'sufficiency_vs_rotation')]:
        pinterval(name, r[key]['estimate'], r[key]['interval'], c)
    percent('TwoRotated', ac['rotated_query']['mean'] / ac['same']['mean'], c)
    number('TwoDQ', ac['same_clamp_q']['mean'], 2, c)
    number('TwoDQueryOnly', ac['same_query_only']['mean'], 2, c)
    for mode, name in [('k', 'TwoKey'), ('v', 'TwoValue'), ('p', 'TwoProb'), ('z', 'TwoOut'), ('other_heads', 'TwoOther')]:
        pinterval(name, post['removal_fraction'][mode]['estimate'], post['removal_fraction'][mode]['ci'], ph)
    pinterval('TwoForeignNec', post['foreign_removal_fraction']['q']['estimate'], post['foreign_removal_fraction']['q']['ci'], ph)
    pinterval('TwoForeignSuf', post['foreign_query_only_fraction']['estimate'], post['foreign_query_only_fraction']['ci'], ph)
    pinterval('TwoBenefitRemoved', post['benefit_removed_by_query_clamp']['estimate'], post['benefit_removed_by_query_clamp']['ci'], ph)
    pinterval('TwoHarmRemoved', post['harm_removed_by_query_clamp']['estimate'], post['harm_removed_by_query_clamp']['ci'], ph)
    pinterval('TwoBenefitRecreated', post['benefit_recreated_by_queries']['estimate'], post['benefit_recreated_by_queries']['ci'], ph)
    pinterval('TwoHarmRecreated', post['harm_recreated_by_queries']['estimate'], post['harm_recreated_by_queries']['ci'], ph)
    cs = conf['condition_summary']
    number('TwoMatched', cs['same_matched']['margin_repair'], 2, c)
    number('TwoOpposite', cs['same_opposite']['margin_repair'], 2, c)
    number('TwoOppositeMag', -cs['same_opposite']['margin_repair'], 2, c)
    number('TwoForeignOppositeMag', -cs['foreign_opposite']['margin_repair'], 2, c)
    number('TwoForeignMatched', cs['foreign_matched']['margin_repair'], 2, c)
    number('TwoForeignOpposite', cs['foreign_opposite']['margin_repair'], 2, c)
    number('TwoSelf', cs['recipient_self']['margin_repair'], 2, c)
    percent('TwoZeroTop', cs['zero']['correct'], c)
    percent('TwoMatchedTop', cs['same_matched']['correct'], c)
    percent('TwoOppositeTop', cs['same_opposite']['correct'], c)
    percent('TwoForeignMatchedTop', cs['foreign_matched']['correct'], c)
    percent('TwoForeignOppositeTop', cs['foreign_opposite']['correct'], c)
    percent('TwoSelfTop', cs['recipient_self']['correct'], c)
    number('TwoAttnMatched', cs['same_matched']['attention_io_minus_s1'], 2, c)
    number('TwoAttnOpposite', cs['same_opposite']['attention_io_minus_s1'], 2, c)
    number('TwoAttnOppositeMag', -cs['same_opposite']['attention_io_minus_s1'], 2, c)
    number('TwoAttnQMatched', cs['same_matched:query_only']['attention_io_minus_s1'], 2, c)
    number('TwoAttnQOpposite', cs['same_opposite:query_only']['attention_io_minus_s1'], 2, c)
    put('TwoRecipients', count(conf['recipients']), c)
    put('TwoSources', count(conf['sources']), c)
    put('TwoDevRecipients', count(dev['recipients']), src + 'development/summary.json')
    put('TwoNewNames', str(summ['confirmation_independent_axes']['fresh_names']), src + 'STUDY-SUMMARY.json')
    put('TwoRecords', count(summ['total_condition_records']), src + 'STUDY-SUMMARY.json')
    put('TwoExamples', count(summ['model_examples']['total']), src + 'STUDY-SUMMARY.json')
    put('TwoConditions', str(len(conf['labels'])), c)
    put('TwoNamePos', str(sum(v > 0 for v in ac['same']['name_means'])), c)
    put('TwoCtxPos', str(sum(v > 0 for v in ac['same']['context_means'])), c)
    put('TwoUnits', str(len(ac['same']['name_means'])), c)
    put('TwoCtx', str(len(ac['same']['context_means'])), c)
    number('TwoDevD', dev['absolute_contrasts']['same']['mean'], 2, src + 'development/summary.json')


# ============================================================================================== Study 3
STRATA = [('third_name@0.5', 'TN', 'H'), ('third_name@1', 'TN', 'F'), ('repeat_io@0.5', 'RR', 'H'), ('repeat_io@1', 'RR', 'F')]


def study3():
    src = 'studies/pathway-interactions-v3/analysis/'
    conf = read(S3 / 'analysis/confirmation/summary.json')
    dev = read(S3 / 'analysis/development/summary.json')
    summ = read(S3 / 'analysis/STUDY-SUMMARY.json')
    rank = read(S3 / 'analysis/SCREEN-RANKING.json')
    post = read(RES / 'posthoc-intervals.json')['study3']
    feed = read(RES / 'query-feed.json')
    c = src + 'confirmation/summary.json'
    ph = 'analysis/results/posthoc-intervals.json'
    for key, corr, dose in STRATA:
        st = conf['strata'][key]
        r = st['ratios']
        p = 'Three' + corr + dose
        interval(p + 'D', st['absolute']['clamp_0']['mean'], *st['absolute']['clamp_0']['ci95'], 2, c)
        pinterval(p + 'JointRem', r['joint_necessity']['estimate'], r['joint_necessity']['interval'], c)
        pinterval(p + 'JointRec', r['joint_sufficiency']['estimate'], r['joint_sufficiency']['interval'], c)
        pinterval(p + 'ExtraRem', r['extra_necessity']['estimate'], r['extra_necessity']['interval'], c)
        pinterval(p + 'ExtraRec', r['extra_sufficiency']['estimate'], r['extra_sufficiency']['interval'], c)
        q = r['clamp_1']
        pinterval(p + 'QRem', 1 - q['estimate'], [1 - q['interval'][1], 1 - q['interval'][0]], c)
        pinterval(p + 'QRec', r['restore_1']['estimate'], r['restore_1']['interval'], c)
        qa = r['clamp_3']
        pinterval(p + 'QARem', 1 - qa['estimate'], [1 - qa['interval'][1], 1 - qa['interval'][0]], c)
        pinterval(p + 'QARec', r['restore_3']['estimate'], r['restore_3']['interval'], c)
        pinterval(p + 'ASpecRem', r['attention_specific_necessity']['estimate'], r['attention_specific_necessity']['interval'], c)
        pinterval(p + 'ASpecRec', r['attention_specific_sufficiency']['estimate'], r['attention_specific_sufficiency']['interval'], c)
        pinterval(p + 'MSpecRem', r['mlp_specific_necessity']['estimate'], r['mlp_specific_necessity']['interval'], c)
        pinterval(p + 'MSpecRec', r['mlp_specific_sufficiency']['estimate'], r['mlp_specific_sufficiency']['interval'], c)
        pinterval(p + 'Inter', r['restore_interaction_total']['estimate'], r['restore_interaction_total']['interval'], c)
        percent(p + 'ARec', r['restore_2']['estimate'], c)
        percent(p + 'MRec', r['restore_4']['estimate'], c)
        put(p + 'Pass', 'yes' if st['primary_gate'] else 'no', c)
        put(p + 'MSpecPass', 'yes' if st['specificity']['mlp'] else 'no', c)
        ps = post[key]
        if corr == 'TN':
            for name, k in [('BenefitRem', 'matched_removed_by_QAM'), ('HarmRem', 'opposite_removed_by_QAM'),
                            ('BenefitRec', 'matched_recreated_by_QAM'), ('HarmRec', 'opposite_recreated_by_QAM')]:
                pinterval(p + name, ps[k]['estimate'], ps[k]['ci'], ph)
        else:
            interval(p + 'OppChange', ps['opposite_change']['estimate'], *ps['opposite_change']['ci95'], 2, ph)
            interval(p + 'MatchedChange', ps['matched_change']['estimate'], *ps['matched_change']['ci95'], 2, ph)
    # ranges over the four strata
    def rng(field, transform=lambda x: x):
        vals = [transform(conf['strata'][k]['ratios'][field]['estimate']) for k, _, _ in STRATA]
        return min(vals), max(vals)
    for name, field in [('ExtraRem', 'extra_necessity'), ('ExtraRec', 'extra_sufficiency'), ('ASpecRem', 'attention_specific_necessity'),
                        ('ASpecRec', 'attention_specific_sufficiency'), ('MSpecRem', 'mlp_specific_necessity'),
                        ('MSpecRec', 'mlp_specific_sufficiency'), ('Inter', 'restore_interaction_total')]:
        lo, hi = rng(field)
        if field == 'restore_interaction_total':
            lo, hi = -hi, -lo                                   # reported as a shortfall below the sum
        put('ThreeRange' + name + 'Lo', pct(lo, 1 if 'MSpec' not in name else 2), c)
        put('ThreeRange' + name + 'Hi', pct(hi, 1 if 'MSpec' not in name else 2), c)
    jn = [conf['strata'][k]['ratios']['joint_necessity']['estimate'] for k, _, _ in STRATA]
    js = [conf['strata'][k]['ratios']['joint_sufficiency']['estimate'] for k, _, _ in STRATA]
    put('ThreeRangeJointRemLo', pct(min(jn)), c)
    put('ThreeRangeJointRemHi', pct(max(jn)), c)
    put('ThreeRangeJointRecLo', pct(min(js)), c)
    put('ThreeRangeJointRecHi', pct(max(js)), c)
    qr = [1 - conf['strata'][k]['ratios']['clamp_1']['estimate'] for k, _, _ in STRATA]
    qs = [conf['strata'][k]['ratios']['restore_1']['estimate'] for k, _, _ in STRATA]
    put('ThreeRangeQRemLo', pct(min(qr)), c)
    put('ThreeRangeQRemHi', pct(max(qr)), c)
    put('ThreeRangeQRecLo', pct(min(qs)), c)
    put('ThreeRangeQRecHi', pct(max(qs)), c)
    tn = conf['strata']['third_name@1']['ratios']
    percent('ThreeTNFRestRem', 1 - tn['joint_necessity']['estimate'], c)     # share of D left after the joint clamp
    percent('ThreeTNFRestRec', 1 - tn['joint_sufficiency']['estimate'], c)   # share of D the joint transplant leaves out
    put('ThreeStrataPassed', str(sum(conf['strata'][k]['primary_gate'] for k, _, _ in STRATA)), c)
    put('ThreeStrataMissed', str(sum(not conf['strata'][k]['primary_gate'] for k, _, _ in STRATA)), c)
    put('ThreeMSpecFailed', str(sum(not conf['strata'][k]['specificity']['mlp'] for k, _, _ in STRATA)), c)
    put('ThreeJointMissed', str(sum(1 for a, b in zip(jn, js) if min(a, b) < .75)), c)
    # development versus confirmation
    dc = summ['development_confirmation_comparison']
    for key, corr, dose in STRATA:
        p = 'Three' + corr + dose
        percent(p + 'DevJointRem', dc[key]['joint_necessity']['development'], src + 'STUDY-SUMMARY.json')
        percent(p + 'DevJointRec', dc[key]['joint_sufficiency']['development'], src + 'STUDY-SUMMARY.json')
        number(p + 'DevD', dev['strata'][key]['absolute']['clamp_0']['mean'], 2, src + 'development/summary.json')
    # selection and screen
    sel = summ['selection']
    put('ThreeCandidates', str(sel['candidate_count']), src + 'STUDY-SUMMARY.json')
    put('ThreeHeads', str(sum(1 for x in rank if x['id'].startswith('A'))), src + 'SCREEN-RANKING.json')
    put('ThreeMLPs', str(sum(1 for x in rank if x['id'].startswith('M'))), src + 'SCREEN-RANKING.json')
    put('ThreeScreenCases', str(sel['screen_recipients']), src + 'STUDY-SUMMARY.json')
    number('ThreeScreenD', sel['full_order_contrast'], 2, src + 'STUDY-SUMMARY.json')
    percent('ThreeScreenQRem', 1 - sel['known_query_clamped_contrast'] / sel['full_order_contrast'], src + 'STUDY-SUMMARY.json')
    percent('ThreeScreenQRest', sel['known_query_clamped_contrast'] / sel['full_order_contrast'], src + 'STUDY-SUMMARY.json')
    byid = {x['id']: (i + 1, x) for i, x in enumerate(rank)}
    for cid, name in [('A15.4', 'ScreenAOne'), ('A12.3', 'ScreenATwo'), ('A14.5', 'ScreenThird'), ('A18.5', 'ScreenFourth'),
                      ('M14', 'ScreenM'), ('A13.13', 'ScreenThirteen'), ('A13.4', 'ScreenThirteenFour')]:
        percent('Three' + name, byid[cid][1]['fraction_of_full'], src + 'SCREEN-RANKING.json')
        put('Three' + name + 'Rank', str(byid[cid][0]), src + 'SCREEN-RANKING.json')
    put('ThreeScreenAboveOne', str(sum(1 for x in rank if x['fraction_of_full'] > .01)), src + 'SCREEN-RANKING.json')
    put('ThreeScreenNegative', str(sum(1 for x in rank if x['fraction_of_full'] < 0)), src + 'SCREEN-RANKING.json')
    percent('ThreeScreenPositiveSum', sum(x['fraction_of_full'] for x in rank if x['fraction_of_full'] > 0), src + 'SCREEN-RANKING.json', d=0)
    # competence and counts
    put('ThreeComp', pct(min(v['top1'] for v in conf['competence'].values()), 2), c)
    put('ThreeRevComp', pct(conf['reverse_corruption_accuracy'], 2), c)
    put('ThreeRecipients', count(conf['recipients']), c)
    put('ThreeDevRecipients', count(dev['recipients']), src + 'development/summary.json')
    put('ThreeConditions', str(conf['conditions']), c)
    put('ThreeRecords', count(summ['total_condition_records']), src + 'STUDY-SUMMARY.json')
    put('ThreeExamples', count(summ['model_examples']['total']), src + 'STUDY-SUMMARY.json')
    cs = conf['strata']['third_name@1']['conditions']
    percent('ThreeZeroTop', cs['zero']['top1'], c)
    number('ThreeMatched', cs['clamp:0:matched@1']['margin_repair'], 2, c)
    percent('ThreeMatchedTop', cs['clamp:0:matched@1']['top1'], c)
    number('ThreeOpposite', cs['clamp:0:opposite@1']['margin_repair'], 2, c)
    number('ThreeOppositeMag', -cs['clamp:0:opposite@1']['margin_repair'], 2, c)
    percent('ThreeOppositeTop', cs['clamp:0:opposite@1']['top1'], c)
    number('ThreeTransMatched', cs['restore:7:matched@1']['margin_repair'], 2, c)
    percent('ThreeTransMatchedTop', cs['restore:7:matched@1']['top1'], c)
    number('ThreeTransOpposite', cs['restore:7:opposite@1']['margin_repair'], 2, c)
    percent('ThreeTransOppositeTop', cs['restore:7:opposite@1']['top1'], c)
    # post hoc trace analysis (query feed)
    fq = 'analysis/results/query-feed.json'
    strata = feed['strata']
    def frange(field, scale=100):
        vals = [s[field] for s in strata.values()]
        return min(vals), max(vals)
    for name, field in [('FeedAHeld', 'queries_surviving_when_A_held'), ('FeedATrans', 'queries_recreated_by_A_transplant'),
                        ('FeedMHeld', 'queries_surviving_when_M_held'), ('FeedMTrans', 'queries_recreated_by_M_transplant')]:
        lo, hi = frange(field)
        put('Three' + name + 'Lo', pct(lo, 0), fq)
        put('Three' + name + 'Hi', pct(hi, 0), fq)
    for head, name in [('14.14', 'FeedFourteen'), ('15.14', 'FeedFifteen')]:
        vals = [s['queries_surviving_when_A_held_by_head'][head] for s in strata.values()]
        put('Three' + name + 'Lo', pct(min(vals), 0), fq)
        put('Three' + name + 'Hi', pct(max(vals), 0), fq)
    vals = [s['added_head_output_surviving_when_Q_held_by_head']['15.4'] for s in strata.values()]
    put('ThreeFeedReverseLo', pct(min(vals), 0), fq)
    put('ThreeFeedReverseHi', pct(max(vals), 0), fq)
    put('ThreeFeedRecipients', str(feed['traced_recipients']), fq)
    put('ThreeFeedPerStratum', str(min(s['recipients'] for s in strata.values())), fq)
    put('ThreeNewNames', str(read(S3 / 'data/inputs.json')['new_confirmation_names']), 'studies/pathway-interactions-v3/data/inputs.json')


# ============================================================================================== by wording
WORDING_ROWS = [('study2_development', 'Studies 2--3 development', 'third name'),
                ('study2_confirmation', 'Study 2 confirmation', 'third name'),
                ('study3_development_repeat_io@1', 'Study 3 development', 'reversed role'),
                ('study3_confirmation_third_name@1', 'Study 3 confirmation', 'third name'),
                ('study3_confirmation_repeat_io@1', 'Study 3 confirmation', 'reversed role')]


def wording():
    """Post hoc order contrast D by wording (analysis/results/posthoc-intervals.json)."""
    w = read(RES / 'posthoc-intervals.json')['order_contrast_by_wording']
    ph = 'analysis/results/posthoc-intervals.json'
    for key, name in [('study2_development', 'WordDev'), ('study2_confirmation', 'WordTwoConf'),
                      ('study3_confirmation_third_name@1', 'WordThreeConf'), ('study3_confirmation_repeat_io@1', 'WordThreeConfRR')]:
        for family, x in w[key].items():
            if family == 'all':
                continue
            label = {'F10': 'Ten', 'F11': 'Eleven', 'F12': 'Twelve', 'F13': 'Thirteen'}[family]
            interval(name + label, x['estimate'], *x['ci95'], 2, ph)
    same = all(abs(w['study2_development'][f]['estimate'] - w['study3_development_third_name@1'][f]['estimate']) < 1e-9
               for f in w['study2_development'])
    assert same, 'Study 3 development (third name, strength 1) reuses Study 2 development'


def table_wording():
    w = read(RES / 'posthoc-intervals.json')['order_contrast_by_wording']
    lines = []
    for key, label, corruption in WORDING_ROWS:
        cells = []
        for family in ['F10', 'F11', 'F12', 'F13', 'all']:
            x = w[key].get(family)
            cells.append('--' if x is None else f"{fmt(x['estimate'], 2)} {{\\scriptsize[{fmt(x['ci95'][0], 2)}, {fmt(x['ci95'][1], 2)}]}}")
        lines.append(' & '.join([label, corruption] + cells) + ' \\\\')
    return '\n'.join(lines) + '\n'


# ============================================================================================== study records
def records():
    """Manifests, replays and the Study 1 single-example replay (appendix on verification)."""
    files = bytes_ = 0
    for study, name in [(S1, 'One'), (S2, 'Two'), (S3, 'Three')]:
        m = read(study / 'EVIDENCE-MANIFEST.json')
        rel = str(study.relative_to(PROJECT)) + '/EVIDENCE-MANIFEST.json'
        put(name + 'ManifestFiles', count(len(m['files'])), rel)
        files += len(m['files'])
        bytes_ += m['total_bytes']
    put('ManifestFiles', count(files), 'studies/*/EVIDENCE-MANIFEST.json')
    number('ManifestGB', bytes_ / 1e9, 1, 'studies/*/EVIDENCE-MANIFEST.json')
    single = read(S1 / 'runs/model-replay/summary.json')
    sr = 'studies/fixed-signal-v1/runs/model-replay/summary.json'
    put('OneSingleExamples', count(single['singleton_model_examples']), sr)
    put('OneSingleTolerance', f"{single['tolerance']:g}", sr)
    number('OneSingleStateErr', max(b['maximum_coordinate_errors']['states'] for b in single['banks'].values()), 4, sr)
    number('OneSingleLogitErr', max(b['maximum_coordinate_errors']['logits'] for b in single['banks'].values()), 4, sr)
    put('OneSingleTopChanges', str(sum(b['top1_changes'] for b in single['banks'].values())), sr)
    sens = read(S1 / 'analysis/REPLAY-SENSITIVITY.json')['banks']
    worst = max(abs(b['changes']['improvement']) for b in sens.values())
    mant, exp = f'{worst:.1e}'.split('e')
    put('OneSensImprovement', f'{mant}\\times 10^{{{int(exp)}}}', 'studies/fixed-signal-v1/analysis/REPLAY-SENSITIVITY.json')
    batch = read(S1 / 'runs/batch-replay/summary.json')
    assert batch['bitwise_identical']
    put('OneBatchExamples', count(batch['model_examples']), 'studies/fixed-signal-v1/runs/batch-replay/summary.json')
    r2 = read(S2 / 'runs/replay/summary.json')
    assert r2['bitwise_identical']
    put('TwoReplayRecords', count(r2['condition_records']), 'studies/position-routing-v2/runs/replay/summary.json')
    put('TwoReplayRecipients', str(r2['recipient_cases']), 'studies/position-routing-v2/runs/replay/summary.json')
    r3 = read(S3 / 'runs/replay/summary.json')
    assert r3['bitwise_identical']
    put('ThreeReplayRecords', count(r3['condition_records']), 'studies/pathway-interactions-v3/runs/replay/summary.json')
    put('ThreeReplayPanels', str(r3['recipient_evaluations']), 'studies/pathway-interactions-v3/runs/replay/summary.json')
    put('OneReplayFiles', str(len(read(S1 / 'analysis/ANALYSIS-REPLAY.json')['files'])), 'studies/fixed-signal-v1/analysis/ANALYSIS-REPLAY.json')
    put('TwoReplayFiles', str(len(read(S2 / 'analysis/ANALYSIS-REPLAY.json')['files'])), 'studies/position-routing-v2/analysis/ANALYSIS-REPLAY.json')


# ============================================================================================== cross-study
def cross():
    freezes = []
    for study, label in [(S1, 'Study 1'), (S2, 'Study 2'), (S3, 'Study 3')]:
        for f in sorted((study / 'protocol').glob('*FREEZE.json')):
            d = read(f)
            freezes.append((d['created_utc'], label, f.stem.replace('-FREEZE', '').replace('-', ' ').lower(), len(d.get('files', {}))))
    freezes.sort()
    first, last = freezes[0][0], freezes[-1][0]
    put('FreezeFirst', first[11:16], 'studies/*/protocol/*FREEZE.json')
    put('FreezeLast', last[11:16], 'studies/*/protocol/*FREEZE.json')
    put('FreezeCount', str(len(freezes)), 'studies/*/protocol/*FREEZE.json')
    gaps = []
    for study in [S1, S2, S3]:
        times = sorted(read(f)['created_utc'] for f in (study / 'protocol').glob('*FREEZE.json'))
        from datetime import datetime
        t = [datetime.fromisoformat(x) for x in times]
        gaps += [(b - a).total_seconds() / 60 for a, b in zip(t, t[1:])]
    put('FreezeGapMax', str(round(max(gaps))), 'studies/*/protocol/*FREEZE.json')
    put('FreezeGapMin', str(round(min(gaps))), 'studies/*/protocol/*FREEZE.json')
    return freezes


# ============================================================================================== tables
def table_freezes(freezes):
    names = {'qualification': 'Inputs, adapter and numerical checks', 'response': 'Response panel and analysis plan',
             'analysis': 'Analysis code and gates', 'confirmation design': 'Confirmation bank design',
             'heldout': 'Selected predictor and held-out analysis', 'audit': 'Audit and replay rules',
             'causal': 'Runner and primary analysis', 'verification': 'Audit and replay sample',
             'design': 'Protocol, inputs, adapter, runner, selection rule, analysis', 'selection': 'Selected sites and controls',
             'confirmation': 'Development expectations; confirmation unchanged'}
    lines = []
    for utc, study, stem, nfiles in freezes:
        lines.append(f'{study} & {utc[11:16]} & {names.get(stem, stem)} & {nfiles} \\\\')
    return '\n'.join(lines) + '\n'


def table_screen():
    rank = read(S3 / 'analysis/SCREEN-RANKING.json')
    cells = [f"{i + 1} & {x['id'][1:] if x['id'].startswith('A') else 'MLP ' + x['id'][1:]} & {fmt(100 * x['fraction_of_full'], 1)}"
             for i, x in enumerate(rank)]
    per = (len(cells) + 2) // 3
    cols = [cells[i * per:(i + 1) * per] for i in range(3)]
    lines = []
    for j in range(per):
        row = [cols[k][j] if j < len(cols[k]) else '& &' for k in range(3)]
        lines.append(' & '.join(row) + ' \\\\')
    return '\n'.join(lines) + '\n'


def table_study3():
    conf = read(S3 / 'analysis/confirmation/summary.json')
    lines = []
    labels = {'third_name@0.5': 'Third name, 0.5', 'third_name@1': 'Third name, 1.0', 'repeat_io@0.5': 'Reversed role, 0.5',
              'repeat_io@1': 'Reversed role, 1.0'}
    for key, _, _ in STRATA:
        st = conf['strata'][key]
        r = st['ratios']
        q, qa = r['clamp_1'], r['clamp_3']
        def p(x, ci):
            return f"{fmt(100 * x, 1)} {{\\scriptsize[{fmt(100 * ci[0], 1)}, {fmt(100 * ci[1], 1)}]}}"
        lines.append(' & '.join([labels[key], fmt(st['absolute']['clamp_0']['mean'], 2),
                                 p(1 - q['estimate'], [1 - q['interval'][1], 1 - q['interval'][0]]),
                                 p(r['joint_necessity']['estimate'], r['joint_necessity']['interval']),
                                 p(r['restore_1']['estimate'], r['restore_1']['interval']),
                                 p(r['joint_sufficiency']['estimate'], r['joint_sufficiency']['interval']),
                                 p(r['restore_interaction_total']['estimate'], r['restore_interaction_total']['interval'])]) + ' \\\\')
    return '\n'.join(lines) + '\n'


def table_factorial():
    """Third-name, full-strength stratum: every prespecified clamp and transplant subset (descriptive 95%)."""
    r = read(S3 / 'analysis/confirmation/summary.json')['strata']['third_name@1']['ratios']
    names = {1: 'Q', 2: 'A', 3: 'Q+A', 4: 'M', 5: 'Q+M', 6: 'A+M', 7: 'Q+A+M'}
    lines = []
    for mask in range(1, 8):
        cl, rs = r[f'clamp_{mask}'], r[f'restore_{mask}']
        lines.append(f"{names[mask]} & {fmt(100 * (1 - cl['estimate']), 1)} {{\\scriptsize[{fmt(100 * (1 - cl['interval'][1]), 1)}, "
                     f"{fmt(100 * (1 - cl['interval'][0]), 1)}]}} & {fmt(100 * rs['estimate'], 1)} {{\\scriptsize[{fmt(100 * rs['interval'][0], 1)}, "
                     f"{fmt(100 * rs['interval'][1], 1)}]}} \\\\")
    for label, key in [('Q + control heads', 'sham_attention'), ('Q + control MLP', 'sham_mlp')]:
        cl, rs = r[f'clamp_{key}'], r[f'restore_{key}']
        lines.append(f"{label} & {fmt(100 * (1 - cl['estimate']), 1)} {{\\scriptsize[{fmt(100 * (1 - cl['interval'][1]), 1)}, "
                     f"{fmt(100 * (1 - cl['interval'][0]), 1)}]}} & {fmt(100 * rs['estimate'], 1)} {{\\scriptsize[{fmt(100 * rs['interval'][0], 1)}, "
                     f"{fmt(100 * rs['interval'][1], 1)}]}} \\\\")
    return '\n'.join(lines) + '\n'


def table_devconf():
    s1 = read(S1 / 'analysis/STUDY-SUMMARY.json')['banks']
    s2d = read(S2 / 'analysis/development/summary.json')
    s2c = read(S2 / 'analysis/confirmation/summary.json')
    dc = read(S3 / 'analysis/STUDY-SUMMARY.json')['development_confirmation_comparison']
    rows = []
    rows.append(('1', 'Gain minus directional error', fmt(s1['holdout']['absolute_improvement'], 3) + '$^a$',
                 fmt(s1['confirmation']['absolute_improvement'], 3)))
    for label, key in [('Retention with different names (\\%)', 'foreign_retention'), ('Removal by query clamp (\\%)', 'query_necessity'),
                       ('Recreation by queries alone (\\%)', 'query_sufficiency')]:
        rows.append(('2', label, fmt(100 * s2d['ratios'][key]['estimate'], 1), fmt(100 * s2c['ratios'][key]['estimate'], 1)))
    labels = {'third_name@0.5': 'third name, 0.5', 'third_name@1': 'third name, 1.0', 'repeat_io@0.5': 'reversed role, 0.5',
              'repeat_io@1': 'reversed role, 1.0'}
    for field, what in [('joint_necessity', 'Joint removal'), ('joint_sufficiency', 'Joint recreation'),
                        ('attention_specific_necessity', 'Added heads, removal over control'),
                        ('mlp_specific_necessity', 'MLP 14, removal over control')]:
        for key, _, _ in STRATA:
            rows.append(('3', f'{what}, {labels[key]} (\\%)', fmt(100 * dc[key][field]['development'], 1),
                         fmt(100 * dc[key][field]['confirmation'], 1)))
    return '\n'.join(f'{a} & {b} & {c} & {d} \\\\' for a, b, c, d in rows) + '\n'


def table_gates():
    s1 = read(S1 / 'analysis/STUDY-SUMMARY.json')['banks']
    s2 = read(S2 / 'analysis/confirmation/summary.json')['gates']
    s3 = read(S3 / 'analysis/confirmation/summary.json')
    yes = lambda b: 'pass' if b else '\\textbf{fail}'
    names1 = {'competence': 'Clean competence', 'positive_interval': 'Interval above zero', 'absolute_improvement': 'Improvement $\\ge 0.05$',
              'relative_improvement': 'Relative improvement $\\ge 10\\%$', 'perpendicular_improvement': 'Perpendicular improvement $\\ge 0.02$',
              'repeatable_names': 'Positive in $\\ge 75\\%$ of name groups', 'repeatable_contexts': 'Positive in $\\ge 75\\%$ of context groups',
              'usable_directions': '$\\ge 95\\%$ usable directions', 'positive_semantic_repair': 'Repair above random'}
    lines = [f'1 & {names1[k]} & {yes(s1["holdout"]["mechanism_gates"][k])} & {yes(s1["confirmation"]["mechanism_gates"][k])} \\\\'
             for k in names1]
    names2 = {'competence': 'Clean competence', 'numerical': 'Numerical qualification', 'positive_positional_contrasts': 'Order contrasts positive',
              'primary_ratios': 'Three primary ratios $\\ge 50\\%$', 'query_specificity': 'Query specificity $\\ge 25\\%$ of $D$',
              'repeatability': 'Contrast positive in $\\ge 6/8$ units and contexts'}
    lines += [f'2 & {names2[k]} & -- & {yes(s2[k])} \\\\' for k in names2]
    names3 = {'recipient_competence': 'Clean competence (recipients)', 'source_competence': 'Clean competence (sources)',
              'reverse_corruption_competence': 'Reversed-role competence', 'numerical': 'Numerical qualification',
              'contrast_consistency': 'Contrast consistency', 'primary_all_robustness_strata': 'Joint $\\ge 75\\%$, extra $\\ge 15\\%$, all strata',
              'specificity_all_strata': 'Added-group specificity $\\ge 5\\%$, all strata'}
    lines += [f'3 & {names3[k]} & -- & {yes(s3["gates"][k])} \\\\' for k in names3]
    return '\n'.join(lines) + '\n'


def render():
    study1()
    study2()
    study3()
    wording()
    records()
    freezes = cross()
    head = ('% Generated by paper/make_numbers.py from saved outputs; do not edit.\n'
            '% Each macro is followed by its source file.\n')
    numbers = head + ''.join(f'\\newcommand{{\\{k}}}{{{v}}} % {SOURCES[k]}\n' for k, v in MACROS.items())
    tables = {'table-freezes.tex': table_freezes(freezes), 'table-screen.tex': table_screen(), 'table-study3.tex': table_study3(),
              'table-factorial.tex': table_factorial(), 'table-devconf.tex': table_devconf(), 'table-gates.tex': table_gates(),
              'table-wording.tex': table_wording()}
    out = {'numbers.tex': numbers}
    out.update({k: '% Generated by paper/make_numbers.py; do not edit.\n' + v for k, v in tables.items()})
    return out


def main():
    out = render()
    check = '--check' in sys.argv
    bad = []
    for name, text in out.items():
        path = GEN / name
        if check:
            if not path.exists() or path.read_text() != text:
                bad.append(name)
        else:
            GEN.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.read_text() != text:
                path.write_text(text)
                print('wrote', path.relative_to(PROJECT))
    if check:
        if bad:
            sys.exit('generated files differ from saved outputs: ' + ', '.join(bad))
        print(f'ok  {len(MACROS)} macros and {len(out) - 1} tables match the saved outputs')
    else:
        print(f'{len(MACROS)} macros, {len(out) - 1} tables')


if __name__ == '__main__':
    main()
