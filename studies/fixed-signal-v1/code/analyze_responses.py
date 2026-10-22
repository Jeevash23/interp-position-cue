"""Full-space response prediction and crossed-group uncertainty."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
import core as c
import response_models as m


def panel(run,output):
    rows=c.read(run/'recipients.json');plans=c.read(run/'conditions.json');sources=c.read(run/'sources.json')
    planmap=defaultdict(list)
    for p in plans:planmap[p['recipient_id']].append(p)
    with np.load(run/'recipients_clean.npz') as z:clean=c.behavior(z['logits'],rows)
    with np.load(run/'recipients_corrupt.npz') as z:bad=c.behavior(z['logits'],rows);badlog=z['logits'].astype(float)
    with np.load(run/'sources_clean.npz') as z:sc=c.behavior(z['logits'],sources)
    with np.load(run/'sources_corrupt.npz') as z:sb=c.behavior(z['logits'],sources)
    source_correct={r['id']:bool(v) for r,v in zip(sources,sc['correct'])}
    response={};behavior=[];qualified={}
    for family in ['F10','F11']:
        for order in [0,1]:
            ix=np.array([r['family']==family and r['order']==order for r in rows]);sep=clean['margin'][ix]-bad['margin'][ix]
            qualified[f'{family}/order{order}']=dict(n=int(ix.sum()),top1=float(clean['correct'][ix].mean()),separation=float(sep.mean()),positive_separation=float((sep>0).mean()))
    source_cells={}
    for order in [0,1]:
        ix=np.array([r['order']==order for r in sources]);sep=sc['margin'][ix]-sb['margin'][ix]
        source_cells[str(order)]=dict(n=int(ix.sum()),top1=float(sc['correct'][ix].mean()),separation=float(sep.mean()),positive_separation=float((sep>0).mean()))
    for ri,row in enumerate(rows):
        pp=planmap[row['id']]
        with np.load(run/f'recipient-{ri:04d}.npz') as z:
            zero=next(i for i,p in enumerate(pp) if p['condition']=='zero_write');states=z['states'].astype(float)-z['states'][zero].astype(float)
            logits=z['logits'];b=c.behavior(logits,[row]*len(pp))
            lp=logits.astype(float)-logsumexp(logits.astype(float),axis=1)[:,None]
            bp=badlog[ri]-logsumexp(badlog[ri]);kl=(np.exp(bp)*(bp-lp)).sum(1)
            for i,p in enumerate(pp):
                info=dict(row_id=row['id'],name=row['name_index'],context=row['context_index'],role=row['role'],cell=f"{row['family']}/order{row['order']}",
                    condition=p['condition'],strength=p['strength'],source=p['source_id'],source_variant=None if p['source_id'] is None else p['source_id'].rsplit(':',1)[1],
                    clean_correct=bool(clean['correct'][ri]),correct=bool(b['correct'][i]),top1=int(b['top1'][i]),margin=float(b['margin'][i]),
                    margin_repair=float(b['margin'][i]-bad['margin'][ri]),io_probability=float(b['io_probability'][i]),entropy=float(b['entropy'][i]),
                    kl_from_corrupted=float(kl[i]),response_norms=np.linalg.norm(states[i],axis=1).tolist(),signal_hash=p['signal_hash'])
                behavior.append(info)
                if p['condition']=='fixed_source':
                    key=(row['name_index'],row['context_index'],row['role'],info['source_variant'],p['strength'],info['cell'])
                    response[key]=(states[i],info,source_correct[p['source_id']])
    pairs=[];xx=[];yy=[]
    for key,(y,info,source_ok) in response.items():
        if key[-1]=='F10/order0':continue
        refkey=(*key[:-1],'F10/order0');x,reference,_=response[refkey]
        xx.append(x);yy.append(y);pairs.append(dict(name=key[0],context=key[1],role=key[2],source_variant=key[3],strength=key[4],cell=key[5],
            gain_key=f'{key[5]}|{key[3]}|{key[4]}',competent=source_ok and info['clean_correct'] and reference['clean_correct']))
    x=np.array(xx);y=np.array(yy)
    c.write(output/'behavior.json',behavior);c.write(output/'pairs.json',pairs)
    np.savez_compressed(output/'response-pairs.npz',x=x,y=y)
    return x,y,pairs,behavior,dict(recipients=qualified,sources=source_cells)


def save_model(output,model):
    c.write(output/'model.json',dict(gains=model['gains'],rank=model['rank'],ridge=model['ridge'],cells=list(model['maps'])))
    np.savez_compressed(output/'model.npz',projection=model['projection'],**{f'map_{i}':v for i,v in enumerate(model['maps'].values())})


def load_model(folder):
    v=c.read(folder/'model.json')
    with np.load(folder/'model.npz') as z:return dict(gains=v['gains'],rank=v['rank'],ridge=v['ridge'],projection=z['projection'],maps={cell:z[f'map_{i}'] for i,cell in enumerate(v['cells'])})


def analyze(run,output,fit_folder=None):
    run,output=Path(run),Path(output);output.mkdir(exist_ok=False);assert c.read(run/'status.json')['status']=='complete'
    x,y,pairs,behavior,competence=panel(run,output)
    keys=[r['gain_key'] for r in pairs];cells=[r['cell'] for r in pairs];names=np.array([r['name'] for r in pairs]);contexts=np.array([r['context'] for r in pairs])
    primary=2;xx=x[:,primary];yy=y[:,primary]
    if fit_folder is None:
        chosen,cv=m.select(xx,yy,keys,cells,names,contexts);model=m.fit(xx,yy,keys,cells,*chosen)
        save_model(output,model);c.write(output/'selection.json',cv)
    else:model=load_model(Path(fit_folder))
    gain=np.array([model['gains'][k] for k in keys])[:,None]*xx;rich=m.predict(model,xx,keys,cells)
    eg,pg=m.errors(xx,yy,gain);er,pr=m.errors(xx,yy,rich);energy=np.sum(yy**2,axis=1)
    norms_x=np.linalg.norm(xx,axis=1);norms_y=np.linalg.norm(yy,axis=1);usable=(norms_x>.25)&(norms_y>.25)
    cosine=np.sum(xx*yy,axis=1)/np.maximum(norms_x*norms_y,1e-30)
    effect=m.crossed_interval(eg-er,energy,names,contexts);perp=m.crossed_interval(pg-pr,energy,names,contexts)
    oracle=yy-(np.sum(xx*yy,axis=1)/np.maximum(np.sum(xx*xx,axis=1),1e-30))[:,None]*xx
    norms=dict(usable=int(usable.sum()),total=len(usable),fraction=float(usable.mean()),minimum_reference=float(norms_x.min()),minimum_target=float(norms_y.min()),
        cosine_mean=float(cosine[usable].mean()) if usable.any() else None,cosine_median=float(np.median(cosine[usable])) if usable.any() else None)
    descriptive={}
    for condition in sorted({r['condition'] for r in behavior}):
        values=[r for r in behavior if r['condition']==condition]
        descriptive[condition]=dict(n=len(values),**{metric:float(np.mean([r[metric] for r in values])) for metric in ['correct','margin_repair','kl_from_corrupted','entropy','io_probability']})
    positive=[r for r in behavior if r['condition']=='fixed_source' and r['strength']>0]
    positive_repair=float(np.mean([r['margin_repair'] for r in positive]));random_repair=descriptive['norm_matched_random']['margin_repair']
    cfg=c.read(c.ROOT/'protocol/config.json')
    competent=all(v['top1']>=cfg['clean_top1_minimum'] and v['separation']>=.5 and v['positive_separation']>=.75 for cells_ in competence.values() for v in cells_.values())
    gainloss=float(eg.sum()/energy.sum());richloss=float(er.sum()/energy.sum())
    gate=dict(competence=competent,positive_interval=effect['ci95'][0]>0,absolute_improvement=effect['mean']>=.05,
        relative_improvement=effect['mean']/gainloss>=.1,perpendicular_improvement=perp['mean']>=.02,
        repeatable_names=sum(v>0 for v in effect['name_effects'])>=.75*len(effect['name_effects']),
        repeatable_contexts=sum(v>0 for v in effect['context_effects'])>=.75*len(effect['context_effects']),
        usable_directions=norms['fraction']>=.95,positive_semantic_repair=positive_repair>0 and positive_repair>random_repair)
    selected=np.array([r['competent'] for r in pairs])
    depth=[]
    for li,layer in enumerate([13,18,23]):
        a=x[:,li];b=y[:,li];n=np.sum(a*a,axis=1);o=b-(np.sum(a*b,axis=1)/np.maximum(n,1e-30))[:,None]*a
        depth.append(dict(block=layer,oracle_scalar_fractional_error=float(np.sum(o*o)/np.sum(b*b)),mean_response_norm=float(np.linalg.norm(b,axis=1).mean())))
    summary=dict(status='complete',scope='development fit; no independent claims' if fit_folder is None else 'held-out conditional prediction',
        recipients=len(c.read(run/'recipients.json')),conditions=len(behavior),prediction_pairs=len(pairs),model_rank=model['rank'],ridge=model['ridge'],
        gain_normalized_error=gainloss,transform_normalized_error=richloss,improvement=effect,perpendicular_improvement=perp,
        oracle_scalar_fractional_error=float(np.sum(oracle*oracle)/energy.sum()),
        projection_energy_fraction=float(np.sum((yy@model['projection'])**2)/energy.sum()),direction_coverage=norms,competence=competence,
        behavior=descriptive,positive_strength_margin_repair=positive_repair,random_margin_repair=random_repair,
        competent_subset=dict(n=int(selected.sum()),total=len(selected),gain_error=float(eg[selected].sum()/energy[selected].sum()),transform_error=float(er[selected].sum()/energy[selected].sum())),
        depth_direction_diagnostics=depth,mechanism_gates=gate,continue_to_mechanism=all(gate.values()),
        analysis_source_sha256=c.sha(__file__),models_source_sha256=c.sha(c.ROOT/'code/response_models.py'))
    np.savez_compressed(output/'prediction-errors.npz',gain=eg,transform=er,perpendicular_gain=pg,perpendicular_transform=pr,energy=energy,cosine=cosine,gain_predictions=gain,transform_predictions=rich)
    c.write(output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--fit')
    a=p.parse_args();analyze(a.run,a.output,a.fit)
