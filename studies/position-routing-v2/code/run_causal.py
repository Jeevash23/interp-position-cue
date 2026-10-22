"""Run the locked 45-condition positional-content and causal-query experiment."""
import argparse
import resource
import time
from pathlib import Path
import numpy as np
import core as c
from mechanism import MechanismEngine,signal_bank,condition_plan


def run(split,output):
    cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/inputs.json')
    rows=[r for r in bundle['recipients'] if r['split']==split];sources=[r for r in bundle['sources'] if r['split']==split]
    output=Path(output);output.mkdir(exist_ok=False);start=time.perf_counter()
    files=[c.ROOT/p for p in ['code/core.py','code/mechanism.py','code/run_causal.py','code/analyze_causal.py','protocol/config.json','protocol/CAUSAL-PLAN.md','data/inputs.json','data/query-rotations.npz']]
    c.write(output/'manifest.json',dict(started_utc=c.now(),split=split,source_hashes=c.snapshot(output,files),planned_recipients=len(rows),planned_conditions=45*len(rows)))
    engine=MechanismEngine(cfg);meta=[];zeroerrors=[]
    try:
        if split=='development':
            assert c.read(c.ROOT/'runs/qualification/summary.json')['status']=='passed'
            baseline={}
            for condition in ['clean','corrupt']:
                with np.load(c.ROOT/f'runs/qualification/sources-{condition}.npz') as z:baseline[condition]={k:z[k] for k in z.files}
            sc,sb=baseline['clean'],baseline['corrupt']
        else:sc=engine.measure(sources,True);sb=engine.measure(sources,False)
        raw,normalized=signal_bank(rows,sources,sc,sb)
        np.savez_compressed(output/'sources-clean.npz',**sc);np.savez_compressed(output/'sources-corrupt.npz',**sb)
        np.savez_compressed(output/'signals.npz',raw=np.stack([raw[r['id']] for r in sources]),normalized=np.stack([normalized[r['id']] for r in sources]))
        c.write(output/'sources.json',sources);c.write(output/'recipients.json',rows)
        hashes={k:dict(raw=c.array_hash(raw[k]),normalized=c.array_hash(normalized[k])) for k in raw};c.write(output/'signal-hashes.json',hashes)
        with np.load(c.ROOT/'data/query-rotations.npz') as z:rot=engine.torch.as_tensor(z['rotations'],device=engine.device)
        zero=np.zeros((12,1024),np.float32)
        for ri,row in enumerate(rows):
            c.guard();plans=condition_plan(row,cfg['consumer_heads'],bundle['control_head_sets'])
            clean,_=engine.probe(row,np.stack([zero]),clean=True)
            natural_deltas=np.stack([zero]+[normalized[row['source_ids'][p['source']]] for p in plans[1:5]])
            natural,cache=engine.probe(row,natural_deltas,capture_baseline=True)
            qbanks={p['id']:engine.torch.as_tensor(natural['q'][i],device=engine.device) for i,p in enumerate(plans[:5])};qbanks['rotations']=rot
            values={k:np.empty((len(plans),*v.shape[1:]),v.dtype) for k,v in natural.items()}
            for k in values:values[k][:5]=natural[k]
            selfdelta=clean['source'][0]-natural['source'][0]
            all_deltas=[]
            for p in plans:
                if p['scale']=='zero':d=zero
                elif p['scale']=='self':d=selfdelta
                elif p['scale']=='raw':d=raw[row['source_ids'][p['source']]]
                elif p['scale']=='normalized':d=normalized[row['source_ids'][p['source']]]
                else:raise ValueError(p)
                all_deltas.append(d)
            for start_index in range(5,len(plans),8):
                pp=plans[start_index:start_index+8];out,_=engine.probe(row,np.stack(all_deltas[start_index:start_index+8]),pp,cache,qbanks)
                for k in values:values[k][start_index:start_index+len(pp)]=out[k]
            zi=next(i for i,p in enumerate(plans) if p['id']=='zero_write')
            ze={k:float(np.max(np.abs(values[k][zi]-values[k][0]))) for k in values};zeroerrors.append(ze)
            assert max(ze.values())<=cfg['numerical_absolute_tolerance']
            for i,p in enumerate(plans):
                record=dict(p,row_id=row['id'],signal_hash=c.array_hash(all_deltas[i]),
                    source_id=None if p['source'] is None else row['source_ids'][p['source']],status='complete')
                if 'bank' in p:record['query_bank_hash']=c.array_hash(natural['q'][next(j for j,x in enumerate(plans[:5]) if x['id']==p['bank'])])
                meta.append(record)
            arrays=dict(values,**{f'clean_{k}':v for k,v in clean.items()},
                baseline_k=np.stack([cache[layer]['k'].cpu().numpy() for layer in cfg['capture_layers']]),
                baseline_v=np.stack([cache[layer]['v'].cpu().numpy() for layer in cfg['capture_layers']]))
            np.savez_compressed(output/f'recipient-{ri:04d}.npz',**arrays)
            c.write(output/'status.json',dict(status='running',completed=ri+1,planned=len(rows),condition_records=(ri+1)*45,model_examples=engine.model_examples))
            if (ri+1)%16==0:print(split,ri+1,'/',len(rows),flush=True)
        assert all(c.array_hash(raw[k])==v['raw'] and c.array_hash(normalized[k])==v['normalized'] for k,v in hashes.items())
        c.write(output/'conditions.json',meta)
        result=dict(status='complete',completed=len(rows),planned=len(rows),condition_records=len(meta),model_examples=engine.model_examples,
            reused_qualification_source_measurements=split=='development',maximum_zero_errors={k:max(v[k] for v in zeroerrors) for k in zeroerrors[0]},
            environment=engine.environment,seconds=time.perf_counter()-start,peak_gpu_bytes=engine.torch.cuda.max_memory_allocated(),
            maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=c.now())
        c.write(output/'status.json',result);print(result,flush=True)
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','confirmation'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.split,a.output)
