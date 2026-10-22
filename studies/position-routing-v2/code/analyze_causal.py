"""Locked positional contrasts and query-path mediation on crossed independent units."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import core as c


def intervals(series,rows,draws=20000):
    ns=sorted({r['name_index'] for r in rows});cs=sorted({r['context_index'] for r in rows})
    counts=np.zeros((len(ns),len(cs)));grids={k:counts.copy() for k in series}
    for i,r in enumerate(rows):
        n,ctx=r['name_index'],r['context_index'];counts[n,ctx]+=1
        for k,v in series.items():grids[k][n,ctx]+=v[i]
    for k in grids:grids[k]/=counts
    rng=np.random.default_rng(2026100814);ni=rng.integers(len(ns),size=(draws,len(ns)));ci=rng.integers(len(cs),size=(draws,len(cs)))
    boot={k:g[ni[:,:,None],ci[:,None,:]].mean((1,2)) for k,g in grids.items()}
    return {k:dict(mean=float(np.mean(v)),ci95=np.quantile(boot[k],[.025,.975]).tolist(),name_means=grids[k].mean(1).tolist(),context_means=grids[k].mean(0).tolist()) for k,v in series.items()},boot


def analyze(run,output):
    run,output=Path(run),Path(output);output.mkdir(exist_ok=False);assert c.read(run/'status.json')['status']=='complete'
    rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');cfg=c.read(c.ROOT/'protocol/config.json');meta=c.read(run/'conditions.json')
    plans=defaultdict(list)
    for p in meta:plans[p['row_id']].append(p)
    labels=[p['id'] for p in plans[rows[0]['id']]];records=[];margins=[];correct=[];clean_records=[];attention=[];source_competence={}
    for kind in ['clean','corrupt']:
        with np.load(run/f'sources-{kind}.npz') as z:source_competence[kind]=c.behavior(z['logits'],sources)
    sourcecells={}
    for kind in ['same','foreign']:
        for order in [0,1]:
            ix=np.array([s['source_kind']==kind and s['order']==order for s in sources]);sep=source_competence['clean']['margin'][ix]-source_competence['corrupt']['margin'][ix]
            sourcecells[f'{kind}/order{order}']=dict(n=int(ix.sum()),top1=float(source_competence['clean']['correct'][ix].mean()),separation=float(sep.mean()),positive_separation=float((sep>0).mean()))
    for i,row in enumerate(rows):
        pp=plans[row['id']];assert [p['id'] for p in pp]==labels
        with np.load(run/f'recipient-{i:04d}.npz') as z:
            b=c.behavior(z['logits'],[row]*len(pp));clean=c.behavior(z['clean_logits'],[row]);margins.append(b['margin']);correct.append(b['correct'])
            lp=z['logits'].astype(float)-logsumexp(z['logits'].astype(float),axis=1)[:,None];kl=np.sum(np.exp(lp[0])*(lp[0]-lp),axis=1)
            at=np.stack([z['p'][:,cfg['capture_layers'].index(l),h] for l,h in cfg['consumer_heads']],axis=1)
            aiodiff=at[:,:,row['io_position']].mean(1)-at[:,:,row['s1_position']].mean(1);attention.append(aiodiff)
            clean_records.append(dict(row_id=row['id'],name=row['name_index'],context=row['context_index'],cell=f"{row['family']}/order{row['order']}",
                correct=bool(clean['correct'][0]),margin=float(clean['margin'][0]),separation=float(clean['margin'][0]-b['margin'][0])))
            for j,p in enumerate(pp):records.append(dict(row_id=row['id'],name=row['name_index'],context=row['context_index'],role=row['role'],cell=f"{row['family']}/order{row['order']}",
                condition=p['id'],margin=float(b['margin'][j]),margin_repair=float(b['margin'][j]-b['margin'][0]),correct=bool(b['correct'][j]),
                top1=int(b['top1'][j]),io_probability=float(b['io_probability'][j]),entropy=float(b['entropy'][j]),kl_from_zero=float(kl[j]),
                attention_io_minus_s1=float(aiodiff[j]),attention_s2=float(at[j,:,row['s2']].mean())))
        if (i+1)%64==0:print('Analyzed',i+1,'/',len(rows),flush=True)
    margins=np.array(margins);correct=np.array(correct);attention=np.array(attention);lookup={label:i for i,label in enumerate(labels)}
    def contrast(kind,suffix=''):return margins[:,lookup[f'{kind}_matched{suffix}']]-margins[:,lookup[f'{kind}_opposite{suffix}']]
    series={'same':contrast('same'),'foreign':contrast('foreign'),'raw_same':contrast('raw_same'),'raw_foreign':contrast('raw_foreign')}
    for kind in ['same','foreign']:
        for mode in ['q','k','v','p','z']:series[f'{kind}_clamp_{mode}']=contrast(kind,f':clamp_{mode}')
        for i in [0,1]:series[f'{kind}_control{i}']=contrast(kind,f':control{i}_q')
        series[f'{kind}_query_only']=contrast(kind,':query_only')
    series['rotated_query']=contrast('same',':rotated_query')
    series['q_removed']=series['same']-series['same_clamp_q']
    series['q_specific_vs_controls']=(series['same_control0']+series['same_control1'])/2-series['same_clamp_q']
    series['q_specific_vs_values']=series['same_clamp_v']-series['same_clamp_q']
    series['q_sufficiency_vs_rotation']=series['same_query_only']-series['rotated_query']
    absolute,boot=intervals(series,rows,cfg['bootstrap_draws']);den=absolute['same']['mean'];denboot=boot['same']
    invalid=int(np.sum(denboot<=.25));ratio_specs=dict(foreign_retention='foreign',query_necessity='q_removed',query_sufficiency='same_query_only',
        specificity_vs_random_heads='q_specific_vs_controls',specificity_vs_values='q_specific_vs_values',sufficiency_vs_rotation='q_sufficiency_vs_rotation')
    ratios={}
    for label,key in ratio_specs.items():
        values=np.divide(boot[key],denboot,out=np.full_like(denboot,np.nan),where=denboot!=0)
        finite=bool(np.isfinite(values).all());primary=label in ['foreign_retention','query_necessity','query_sufficiency'];alpha=.05/3 if primary else .05
        ratios[label]=dict(estimate=float(absolute[key]['mean']/den) if den!=0 else None,
            interval=np.quantile(values,[alpha/2,1-alpha/2]).tolist() if finite else None,interval_level=1-alpha,
            bootstrap_denominators_below_floor=invalid,bootstrap_draws=len(denboot),interpretable=den>.25 and invalid<=.01*len(denboot))
    rec_competence={}
    for cell in sorted({r['cell'] for r in clean_records}):
        selected=[r for r in clean_records if r['cell']==cell]
        rec_competence[cell]=dict(n=len(selected),top1=float(np.mean([r['correct'] for r in selected])),separation=float(np.mean([r['separation'] for r in selected])),positive_separation=float(np.mean([r['separation']>0 for r in selected])))
    competence=all(v['top1']>=.85 and v['separation']>=.5 and v['positive_separation']>=.75 for v in list(sourcecells.values())+list(rec_competence.values()))
    primary_pass=all(ratios[k]['interpretable'] and ratios[k]['estimate']>=.5 and ratios[k]['interval'] is not None and ratios[k]['interval'][0]>0 for k in ['foreign_retention','query_necessity','query_sufficiency'])
    specific=all(ratios[k]['interpretable'] and ratios[k]['estimate']>=.25 and ratios[k]['interval'] is not None and ratios[k]['interval'][0]>0 for k in ['specificity_vs_random_heads','specificity_vs_values','sufficiency_vs_rotation'])
    consistency=sum(v>0 for v in absolute['same']['name_means'])>=.75*len(absolute['same']['name_means']) and sum(v>0 for v in absolute['same']['context_means'])>=.75*len(absolute['same']['context_means'])
    gates=dict(numerical=c.read(c.ROOT/'runs/qualification/summary.json')['status']=='passed',competence=competence,
        positive_positional_contrasts=den>.25 and absolute['foreign']['mean']>0,primary_ratios=primary_pass,query_specificity=specific,repeatability=consistency)
    condition_summary={label:{metric:float(np.mean([r[metric] for r in records if r['condition']==label])) for metric in ['margin_repair','correct','io_probability','entropy','kl_from_zero','attention_io_minus_s1','attention_s2']} for label in labels}
    c.write(output/'behavior.json',records);c.write(output/'clean-behavior.json',clean_records)
    np.savez_compressed(output/'paired-scores.npz',margins=margins,correct=correct,attention_io_minus_s1=attention,**{f'contrast_{k}':v for k,v in series.items()})
    summary=dict(status='complete',scope='independent confirmation' if rows[0]['split']=='confirmation' else 'development engineering/descriptive check',
        recipients=len(rows),sources=len(sources),condition_records=len(records),labels=labels,absolute_contrasts=absolute,ratios=ratios,
        competence=dict(recipients=rec_competence,sources=sourcecells),gates=gates,bounded_explanation_supported=all(gates.values()),
        condition_summary=condition_summary,source_sha256=c.sha(__file__))
    c.write(output/'summary.json',summary);print('Causal analysis complete; gates:',gates,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.run,a.output)
