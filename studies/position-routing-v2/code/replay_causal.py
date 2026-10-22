"""Repeat fixed corner blocks with the original batch shape and cached source vectors."""
from collections import defaultdict
import time
import numpy as np
import core as c
from mechanism import MechanismEngine


def main():
    output=c.ROOT/'runs/replay';output.mkdir(exist_ok=False);cfg=c.read(c.ROOT/'protocol/config.json');engine=MechanismEngine(cfg)
    start=time.perf_counter();checks=[]
    def compare(bank,label,new,old):
        err={k:float(np.max(np.abs(new[k]-old[k]))) for k in new}
        checks.append(dict(bank=bank,label=label,errors=err,bitwise_equal=all(np.array_equal(new[k],old[k]) for k in new),
            new_hashes={k:c.array_hash(v) for k,v in new.items()},old_hashes={k:c.array_hash(v) for k,v in old.items()}))
    try:
        for bank in ['development','confirmation']:
            run=c.ROOT/'runs'/bank;rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json')
            plans=defaultdict(list)
            for p in c.read(run/'conditions.json'):plans[p['row_id']].append(p)
            for clean in [True,False]:
                new=engine.measure(sources,clean)
                with np.load(run/f'sources-{"clean" if clean else "corrupt"}.npz') as z:old={k:z[k] for k in z.files}
                compare(bank,f'sources_{clean}',new,old)
            with np.load(run/'signals.npz') as z:raw={r['id']:d for r,d in zip(sources,z['raw'])};normalized={r['id']:d for r,d in zip(sources,z['normalized'])}
            with np.load(c.ROOT/'data/query-rotations.npz') as z:rot=engine.torch.as_tensor(z['rotations'],device=engine.device)
            zero=np.zeros((12,1024),np.float32);nmax=max(r['name_index'] for r in rows);cmax=max(r['context_index'] for r in rows)
            selected=[i for i,r in enumerate(rows) if r['name_index'] in [0,nmax] and r['context_index'] in [0,cmax]]
            assert len(selected)==32
            for ri in selected:
                row=rows[ri];pp=plans[row['id']];clean,_=engine.probe(row,np.stack([zero]),clean=True)
                natural,cache=engine.probe(row,np.stack([zero]+[normalized[row['source_ids'][p['source']]] for p in pp[1:5]]),capture_baseline=True)
                qbanks={p['id']:engine.torch.as_tensor(natural['q'][i],device=engine.device) for i,p in enumerate(pp[:5])};qbanks['rotations']=rot
                values={k:np.empty((45,*v.shape[1:]),v.dtype) for k,v in natural.items()}
                for k in values:values[k][:5]=natural[k]
                selfdelta=clean['source'][0]-natural['source'][0];deltas=[]
                for p in pp:
                    if p['scale']=='zero':d=zero
                    elif p['scale']=='self':d=selfdelta
                    elif p['scale']=='raw':d=raw[row['source_ids'][p['source']]]
                    else:d=normalized[row['source_ids'][p['source']]]
                    assert c.array_hash(d)==p['signal_hash'];deltas.append(d)
                for start_index in range(5,45,8):
                    out,_=engine.probe(row,np.stack(deltas[start_index:start_index+8]),pp[start_index:start_index+8],cache,qbanks)
                    for k in values:values[k][start_index:start_index+8]=out[k]
                new=dict(values,**{f'clean_{k}':v for k,v in clean.items()},baseline_k=np.stack([cache[l]['k'].cpu().numpy() for l in cfg['capture_layers']]),baseline_v=np.stack([cache[l]['v'].cpu().numpy() for l in cfg['capture_layers']]))
                with np.load(run/f'recipient-{ri:04d}.npz') as z:old={k:z[k] for k in z.files}
                compare(bank,row['id'],new,old)
            print(bank,'replayed 32 complete recipients',flush=True)
        exact=all(v['bitwise_equal'] for v in checks);maximum=max(v for r in checks for v in r['errors'].values())
        c.write(output/'summary.json',dict(status='passed' if maximum<=cfg['numerical_absolute_tolerance'] else 'failed',bitwise_identical=exact,
            maximum_coordinate_error=maximum,model_examples=engine.model_examples,recipient_cases=64,condition_records=64*45,
            source_pairs_reextracted=96,seconds=time.perf_counter()-start,finished_utc=c.now(),checks=checks,environment=engine.environment))
        assert maximum<=cfg['numerical_absolute_tolerance']
    finally:engine.close()


if __name__=='__main__':main()
