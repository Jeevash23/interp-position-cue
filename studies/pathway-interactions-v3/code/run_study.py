"""Measure a complete screen or factorial panel with compact primary evidence and fixed full traces."""
import argparse,time,resource
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import core as c
from path_engine import PathEngine,source_bank,screen_plan,factorial_plan,natural_label


def audit_row(row,rows):
    return (row['name_index'] in [min(r['name_index'] for r in rows),max(r['name_index'] for r in rows)] and
            row['context_index'] in [min(r['context_index'] for r in rows),max(r['context_index'] for r in rows)] and row['role']==0 and row['order']==0)


def measure_row(engine,row,signals,plan,full_trace=False):
    cfg=engine.config;zero=np.zeros((12,1024),np.float32)
    labels=['zero']+[natural_label(align,dose) for dose in cfg['strengths'] for align in ['matched','opposite']]
    deltas=[zero]+[signals[row['source_ids'][align]]*dose for dose in cfg['strengths'] for align in ['matched','opposite']]
    clean=engine.probe(row,np.stack([zero]),clean=True)
    natural=engine.probe(row,np.stack(deltas));banks=engine.banks(natural,labels)
    base=natural['logits'][0].astype(float);base_lp=base-logsumexp(base);base_p=np.exp(base_lp)
    values={};hashes=[];full=[];traces={k:[] for k in ['q','z','m']}
    for start in range(0,len(plan),8):
        pp=plan[start:start+8];dd=[]
        for p in pp:
            d=zero if p['kind'] in ['zero','restore'] else signals[row['source_ids'][p['align']]]*p['dose']
            dd.append(d);hashes.append(c.array_hash(d))
        out=engine.probe(row,np.stack(dd),pp,banks)
        z=out['logits'].astype(float);lz=logsumexp(z,axis=1);lp=z-lz[:,None]
        current=dict(target_logits=out['logits'][:,[row['io_id'],row['subject_id']]],top1=z.argmax(1),log_normalizer=lz,
            entropy=-np.sum(np.exp(lp)*lp,axis=1),kl_from_zero=np.sum(base_p[None]*(base_lp[None]-lp),axis=1),
            state=out['state'],write_error=out['write_error'])
        for k,v in current.items():values.setdefault(k,[]).append(v)
        if full_trace:
            full.append(out['logits'])
            for k in traces:traces[k].append(out[k])
    result={k:np.concatenate(v) for k,v in values.items()}
    result.update(clean_target_logits=clean['logits'][0,[row['io_id'],row['subject_id']]],clean_top1=clean['logits'][0].argmax(),
        natural_logits=natural['logits'],natural_state=natural['state'],clean_source=clean['source'][0],
        signal_hashes=np.array(hashes),**{f'natural_{k}':natural[k] for k in ['q','z','m']})
    if full_trace:result.update(logits=np.concatenate(full),**{f'trace_{k}':np.concatenate(v) for k,v in traces.items()})
    assert np.array_equal(result['target_logits'][0],result['target_logits'][1])
    assert np.array_equal(result['state'][0],result['state'][1]) and result['write_error'].max()==0
    errors={}
    for i,p in enumerate(plan):
        if p.get('mask')==-2:
            j=0 if p['kind']=='clamp' else labels.index(natural_label(p['align'],p['dose']))
            errors[p['id']]=max(float(np.max(np.abs(result['target_logits'][i]-natural['logits'][j,[row['io_id'],row['subject_id']]]))),
                               float(np.max(np.abs(result['state'][i]-natural['state'][j]))))
    assert not errors or max(errors.values())<=cfg['numerical_absolute_tolerance']
    return result,errors


def run(phase,output):
    cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/inputs.json')
    split='development' if phase=='screen' else phase
    allrows=[r for r in bundle['recipients'] if r['split']==split];rows=allrows if phase!='screen' else [r for r in allrows if r['id'] in bundle['screen_ids']]
    sources=[r for r in bundle['sources'] if r['split']==split]
    selection=None if phase=='screen' else c.read(c.ROOT/'analysis/SELECTION.json')
    plan=screen_plan(cfg) if phase=='screen' else factorial_plan(selection,cfg)
    output=Path(output);output.mkdir(exist_ok=False)
    files=[c.ROOT/p for p in ['code/core.py','code/path_engine.py','code/run_study.py','code/analyze.py','code/select.py','protocol/config.json','protocol/PLAN.md','data/inputs.json']]
    if selection:files.append(c.ROOT/'analysis/SELECTION.json')
    c.write(output/'manifest.json',dict(started_utc=c.now(),phase=phase,source_hashes=c.snapshot(output,files),planned_recipients=len(rows),planned_conditions=len(rows)*len(plan)))
    c.write(output/'recipients.json',rows);c.write(output/'sources.json',sources);c.write(output/'conditions.json',plan)
    engine=PathEngine(cfg);start=time.perf_counter();maxerror=0.;traceids=[]
    try:
        if split=='development':
            assert c.read(c.ROOT/'runs/qualification/summary.json')['status']=='passed'
            def readarrays(p):
                with np.load(p) as z:return {k:z[k] for k in z.files}
            sc=readarrays(c.ROOT/'runs/qualification/sources-clean.npz');sb=readarrays(c.ROOT/'runs/qualification/sources-corrupt.npz')
        else:sc=engine.measure(sources,True);sb=engine.measure(sources,False)
        raw,signals=source_bank(allrows,sources,sc,sb)
        for label,v in [('clean',sc),('corrupt',sb)]:np.savez_compressed(output/f'sources-{label}.npz',**v)
        np.savez_compressed(output/'signals.npz',raw=np.stack([raw[r['id']] for r in sources]),normalized=np.stack([signals[r['id']] for r in sources]))
        c.write(output/'signal-hashes.json',{sid:c.array_hash(v) for sid,v in signals.items()})
        for i,row in enumerate(rows):
            c.guard();trace=audit_row(row,rows)
            result,errors=measure_row(engine,row,signals,plan,trace)
            if trace:traceids.append(row['id'])
            maxerror=max(maxerror,max(errors.values(),default=0.))
            np.savez_compressed(output/f'row-{i:04d}.npz',**result)
            c.write(output/'status.json',dict(status='running',completed=i+1,planned=len(rows),condition_records=(i+1)*len(plan),model_examples=engine.model_examples))
            if (i+1)%8==0:print(phase,i+1,'/',len(rows),flush=True)
        c.write(output/'status.json',dict(status='complete',completed=len(rows),planned=len(rows),condition_records=len(rows)*len(plan),model_examples=engine.model_examples,
            reused_qualification_sources=split=='development',full_trace_rows=traceids,complete_path_max_error=maxerror,seconds=time.perf_counter()-start,
            environment=engine.environment,peak_gpu_bytes=engine.torch.cuda.max_memory_allocated(),maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=c.now()))
        print(phase,'complete;',engine.model_examples,'physical model examples',flush=True)
    finally:engine.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','development','confirmation'],required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.phase,a.output)
