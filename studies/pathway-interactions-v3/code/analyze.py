"""Frozen factorial contrasts, robustness strata and crossed inference."""
import argparse
from pathlib import Path
import numpy as np
import core as c
from path_engine import natural_label


def crossed(series,rows,draws,seed):
    ns=sorted({r['name_index'] for r in rows});cs=sorted({r['context_index'] for r in rows})
    counts=np.zeros((len(ns),len(cs)));grids={k:counts.copy() for k in series}
    for i,r in enumerate(rows):
        n=ns.index(r['name_index']);ctx=cs.index(r['context_index']);counts[n,ctx]+=1
        for k,v in series.items():grids[k][n,ctx]+=v[i]
    assert np.all(counts>0)
    for k in grids:grids[k]/=counts
    rng=np.random.default_rng(seed);ni=rng.integers(len(ns),size=(draws,len(ns)));ci=rng.integers(len(cs),size=(draws,len(cs)))
    boot={k:g[ni[:,:,None],ci[:,None,:]].mean((1,2)) for k,g in grids.items()}
    observed={k:dict(mean=float(g.mean()),ci95=np.quantile(boot[k],[.025,.975]).tolist(),name_means=g.mean(1).tolist(),context_means=g.mean(0).tolist()) for k,g in grids.items()}
    return observed,boot,grids


def analyze(run,output):
    run,output=Path(run),Path(output);assert c.read(run/'status.json')['status']=='complete';output.mkdir(exist_ok=False)
    cfg=c.read(c.ROOT/'protocol/config.json');rows=c.read(run/'recipients.json');plan=c.read(run/'conditions.json');ix={p['id']:i for i,p in enumerate(plan)}
    margins=[];top=[];clean=[];ent=[];kl=[];probs=[]
    for i,row in enumerate(rows):
        with np.load(run/f'row-{i:04d}.npz') as z:
            t=z['target_logits'].astype(float);m=t[:,0]-t[:,1];margins.append(m);top.append(z['top1']==row['io_id']);ent.append(z['entropy']);kl.append(z['kl_from_zero']);probs.append(np.exp(t[:,0]-z['log_normalizer']))
            cm=float(z['clean_target_logits'][0])-float(z['clean_target_logits'][1]);clean.append(dict(correct=bool(z['clean_top1']==row['io_id']),separation=cm-m[0],corrupt_reverse_correct=bool(z['top1'][0]==row['subject_id'])))
    margins=np.array(margins);top=np.array(top);ent=np.array(ent);kl=np.array(kl);probs=np.array(probs);strata={};grids_saved={}
    for corruption in ['third_name','repeat_io']:
        take=[i for i,r in enumerate(rows) if r['corruption']==corruption];rr=[rows[i] for i in take];mm=margins[take]
        for dose in cfg['strengths']:
            key=f'{corruption}@{dose:g}'
            def d(kind,mask):return mm[:,ix[f'{kind}:{mask}:{natural_label("matched",dose)}']]-mm[:,ix[f'{kind}:{mask}:{natural_label("opposite",dose)}']]
            series={f'clamp_{mask}':d('clamp',mask) for mask in range(8)}
            series.update({f'restore_{mask}':d('restore',mask) for mask in range(1,8)})
            for kind in ['clamp','restore']:
                for control in ['sham_attention','sham_mlp','all']:series[f'{kind}_{control}']=d(kind,control)
            D=series['clamp_0'];series['joint_necessity']=D-series['clamp_7'];series['joint_sufficiency']=series['restore_7']
            series['extra_necessity']=series['clamp_1']-series['clamp_7'];series['extra_sufficiency']=series['restore_7']-series['restore_1']
            for bit,name in [(2,'attention'),(4,'mlp')]:
                series[f'{name}_specific_necessity']=series[f'clamp_sham_{name}']-series[f'clamp_{1|bit}']
                series[f'{name}_specific_sufficiency']=series[f'restore_{1|bit}']-series[f'restore_sham_{name}']
            for a,b in [(1,2),(1,4),(2,4)]:
                series[f'restore_interaction_{a}_{b}']=series[f'restore_{a|b}']-series[f'restore_{a}']-series[f'restore_{b}']
                series[f'clamp_interaction_{a}_{b}']=series[f'clamp_{a}']+series[f'clamp_{b}']-series[f'clamp_{a|b}']-D
            series['restore_interaction_total']=series['restore_7']-series['restore_1']-series['restore_2']-series['restore_4']
            absolute,boot,grids=crossed(series,rr,cfg['bootstrap_draws'],cfg['bootstrap_seed']);den=absolute['clamp_0']['mean'];denboot=boot['clamp_0'];ratios={}
            primary=['joint_necessity','joint_sufficiency','extra_necessity','extra_sufficiency']
            for metric in [k for k in series if k!='clamp_0']:
                values=np.divide(boot[metric],denboot,out=np.full_like(denboot,np.nan),where=denboot!=0);alpha=.05/16 if metric in primary else .05
                finite=bool(np.isfinite(values).all())
                ratios[metric]=dict(estimate=absolute[metric]['mean']/den if den!=0 else None,interval=np.quantile(values,[alpha/2,1-alpha/2]).tolist() if finite else None,interval_level=1-alpha,
                    invalid_denominators=int(np.sum(denboot<=.25)),interpretable=finite and den>.25 and np.sum(denboot<=.25)<=.01*len(denboot))
            primary_pass=all(ratios[k]['interpretable'] and ratios[k]['estimate']>=(.75 if k.startswith('joint') else .15) and ratios[k]['interval'][0]>(.5 if k.startswith('joint') else 0) for k in primary)
            consistency=sum(v>0 for v in absolute['clamp_0']['name_means'])>=.75*len(absolute['clamp_0']['name_means']) and sum(v>0 for v in absolute['clamp_0']['context_means'])>=.75*len(absolute['clamp_0']['context_means'])
            specific={name:all(ratios[f'{name}_specific_{kind}']['interpretable'] and ratios[f'{name}_specific_{kind}']['estimate']>=.05 and ratios[f'{name}_specific_{kind}']['interval'][0]>0 for kind in ['necessity','sufficiency']) for name in ['attention','mlp']}
            cond={p['id']:dict(margin_repair=float((mm[:,j]-mm[:,0]).mean()),top1=float(top[take,j].mean()),io_probability=float(probs[take,j].mean()),entropy=float(ent[take,j].mean()),kl_from_zero=float(kl[take,j].mean())) for j,p in enumerate(plan) if p['dose'] in [0,dose]}
            strata[key]=dict(absolute=absolute,ratios=ratios,primary_gate=primary_pass,consistency=consistency,specificity=specific,conditions=cond)
            for k,g in grids.items():grids_saved[f'{key}:{k}']=g
    competence={}
    for family in sorted({r['family'] for r in rows}):
        for order in [0,1]:
            ii=[i for i,r in enumerate(rows) if r['family']==family and r['order']==order]
            competence[f'{family}/order{order}']=dict(n=len(ii),top1=float(np.mean([clean[i]['correct'] for i in ii])),separation=float(np.mean([clean[i]['separation'] for i in ii])),positive_separation=float(np.mean([clean[i]['separation']>0 for i in ii])))
    cleanpass=all(v['top1']>=.85 and v['separation']>=.5 and v['positive_separation']>=.75 for v in competence.values())
    reverse=[i for i,r in enumerate(rows) if r['corruption']=='repeat_io'];reverse_accuracy=float(np.mean([clean[i]['corrupt_reverse_correct'] for i in reverse]))
    source_comp={}
    with np.load(run/'sources-clean.npz') as z:sc=z['logits']
    with np.load(run/'sources-corrupt.npz') as z:sb=z['logits']
    sources=c.read(run/'sources.json');a=c.behavior(sc,sources);b=c.behavior(sb,sources)
    for corruption in ['third_name','repeat_io']:
        for order in [0,1]:
            ii=np.array([s['corruption']==corruption and s['order']==order for s in sources]);sep=a['margin'][ii]-b['margin'][ii]
            source_comp[f'{corruption}/{order}']=dict(top1=float(a['correct'][ii].mean()),separation=float(sep.mean()),positive_separation=float((sep>0).mean()))
    sourcepass=all(v['top1']>=.85 and v['separation']>=.5 and v['positive_separation']>=.75 for v in source_comp.values())
    gates=dict(numerical=c.read(c.ROOT/'runs/qualification/summary.json')['status']=='passed',recipient_competence=cleanpass,source_competence=sourcepass,
        reverse_corruption_competence=reverse_accuracy>=.85,primary_all_robustness_strata=all(v['primary_gate'] for v in strata.values()),
        contrast_consistency=all(v['consistency'] for v in strata.values()),specificity_all_strata=all(all(v['specificity'].values()) for v in strata.values()))
    np.savez_compressed(output/'paired-scores.npz',margins=margins,top1=top,**grids_saved)
    c.write(output/'summary.json',dict(scope='independent confirmation' if rows[0]['split']=='confirmation' else 'development descriptive after selection',
        recipients=len(rows),conditions=len(plan),condition_records=len(rows)*len(plan),competence=competence,source_competence=source_comp,
        reverse_corruption_accuracy=reverse_accuracy,strata=strata,gates=gates,strong_explanation_supported=all(gates.values()),source_sha256=c.sha(__file__)))
    print('Analysis complete:',gates,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.run,a.output)
