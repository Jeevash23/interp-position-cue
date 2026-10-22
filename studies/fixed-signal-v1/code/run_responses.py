"""Cross cached source corrections and controls through every recipient."""
import argparse
import json
from pathlib import Path
import resource
import time
import numpy as np
import core as c


def run(split,output):
    cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/pilot-qualified.json')
    sources=[r for r in bundle['sources'] if r['split']==split];rows=[r for r in bundle['recipients'] if r['split']==split]
    plans=[p for p in bundle['plans'] if p['split']==split];output=Path(output);output.mkdir(exist_ok=False)
    files=[Path(__file__),c.ROOT/'code/core.py',c.ROOT/'data/pilot-qualified.json',c.ROOT/'protocol/config.json',c.ROOT/'protocol/RESPONSE-ANALYSIS.md',c.ROOT/'code/response_models.py']
    hashes=c.snapshot(output,files);c.write(output/'manifest.json',dict(started_utc=c.now(),split=split,source_hashes=hashes,recipients=len(rows),conditions=len(plans)))
    engine=c.Engine(cfg);start=time.perf_counter();done=[]
    try:
        if split=='development':
            q=c.ROOT/'runs/qualification';assert c.read(q/'summary.json')['competence_passed']
            baseline={}
            for kind in ['sources','recipients']:
                for condition in ['clean','corrupt']:
                    with np.load(q/f'{kind}-{condition}.npz') as z:baseline[f'{kind}_{condition}']={k:z[k] for k in z.files}
        else:
            baseline={f'{kind}_{condition}':engine.measure(rr,condition=='clean') for kind,rr in [('sources',sources),('recipients',rows)] for condition in ['clean','corrupt']}
        for key,v in baseline.items():np.savez_compressed(output/f'{key}.npz',**v)
        source_delta=baseline['sources_clean']['source']-baseline['sources_corrupt']['source']
        cache={r['id']:d for r,d in zip(sources,source_delta)};cachehash={k:c.array_hash(v) for k,v in cache.items()}
        np.savez_compressed(output/'source-cache.npz',deltas=source_delta);c.write(output/'sources.json',sources);c.write(output/'recipients.json',rows)
        random={};meta=[];largest_zero=0.
        for i,row in enumerate(rows):
            c.guard();conditions=[p for p in plans if p['recipient_id']==row['id']];active=[];deltas=[]
            for plan in conditions:
                plan=dict(plan);condition=plan['condition']
                if condition=='untouched_corrupted':delta=None
                elif condition=='zero_write':delta=np.zeros((12,1024),np.float32)
                elif condition=='recipient_self':delta=baseline['recipients_clean']['source'][i]-baseline['recipients_corrupt']['source'][i]
                elif condition=='fixed_source':delta=cache[plan['source_id']]*np.float32(plan['strength'])
                elif condition=='norm_matched_random':
                    draw=plan['random_draw_id']
                    if draw not in random:
                        a=np.random.default_rng(plan['random_seed']).normal(size=(12,1024)).astype(np.float32)
                        a*=np.linalg.norm(cache[plan['source_id']],axis=1,keepdims=True)/np.linalg.norm(a,axis=1,keepdims=True)
                        random[draw]=a
                    delta=random[draw]
                else:raise ValueError(condition)
                plan['signal_hash']=None if delta is None else c.array_hash(delta)
                plan['unscaled_source_hash']=cachehash.get(plan.get('source_id'))
                plan['status']='complete';meta.append(plan)
                if delta is not None:active.append(plan);deltas.append(delta)
            measured=engine.measure([row]*len(active),deltas=np.stack(deltas))
            values={k:np.empty((len(conditions),*a.shape[1:]),a.dtype) for k,a in measured.items()}
            j=0
            for ci,plan in enumerate(conditions):
                if plan['condition']=='untouched_corrupted':
                    for k in values:values[k][ci]=baseline['recipients_corrupt'][k][i]
                else:
                    for k in values:values[k][ci]=measured[k][j]
                    j+=1
            zi=next(j for j,p in enumerate(conditions) if p['condition']=='zero_write')
            error=float(np.max(np.abs(values['logits'][zi]-baseline['recipients_corrupt']['logits'][i])))
            largest_zero=max(largest_zero,error);assert error<cfg['numerical_absolute_tolerance']
            np.savez_compressed(output/f'recipient-{i:04d}.npz',**values)
            done.append(row['id']);c.write(output/'status.json',dict(status='running',completed=len(done),planned=len(rows),failures=[]))
            if (i+1)%16==0:print(json.dumps(dict(split=split,completed=i+1,total=len(rows))),flush=True)
        for k,d in cache.items():assert c.array_hash(d)==cachehash[k]
        c.write(output/'conditions.json',meta);c.write(output/'source-hashes.json',cachehash)
        np.savez_compressed(output/'random-cache.npz',**random)
        summary=dict(status='complete',completed=len(done),planned=len(rows),conditions=len(meta),failures=[],
            maximum_zero_write_logit_error=largest_zero,source_cache_hashes=cachehash,environment=engine.environment,
            seconds=time.perf_counter()-start,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_peak_allocated_bytes=engine.torch.cuda.max_memory_allocated(),finished_utc=c.now(),
            reused_development_qualification_baselines=split=='development')
        c.write(output/'status.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='source_cache_hashes'},indent=2),flush=True)
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split',choices=['development','holdout'],required=True);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.split,a.output)
