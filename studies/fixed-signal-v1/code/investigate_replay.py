"""Post-audit investigation: repeat original batch shapes without changing tolerances."""
from collections import defaultdict
import numpy as np
import core as c


def main():
    output=c.ROOT/'runs/batch-replay';output.mkdir(exist_ok=False)
    cfg=c.read(c.ROOT/'protocol/config.json');engine=c.Engine(cfg);total=0;checks=[]
    c.snapshot(output,[c.ROOT/'code/investigate_replay.py',c.ROOT/'code/core.py',c.ROOT/'protocol/config.json'])
    try:
        for bank in ['development','holdout','confirmation']:
            run=c.ROOT/'runs'/bank;rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json')
            sample=c.read(c.ROOT/'runs/model-replay'/bank/'summary.json')['recipient_indices'];baseline={}
            def compare(label,new,old):
                checks.append(dict(bank=bank,label=label,examples=len(new['logits']),
                    coordinate_errors={k:float(np.max(np.abs(new[k]-old[k]))) for k in new},
                    old_hashes={k:c.array_hash(v) for k,v in old.items()},new_hashes={k:c.array_hash(v) for k,v in new.items()},
                    top1_changes=int(np.sum(new['logits'].argmax(1)!=old['logits'].argmax(1)))))
            for kind,rr in [('sources',sources),('recipients',rows)]:
                for clean in [True,False]:
                    label=f'{kind}_{"clean" if clean else "corrupt"}';new=engine.measure(rr,clean);total+=len(rr)
                    with np.load(run/f'{label}.npz') as z:old={k:z[k] for k in z.files}
                    compare(label,new,old)
                    if kind=='recipients':baseline[label]=old['source']
            with np.load(run/'source-cache.npz') as z:cache={r['id']:d for r,d in zip(sources,z['deltas'])}
            with np.load(run/'random-cache.npz') as z:random={k:z[k] for k in z.files}
            plans=defaultdict(list)
            for p in c.read(run/'conditions.json'):plans[p['recipient_id']].append(p)
            for index in sample:
                row=rows[index];active=[];deltas=[]
                for i,p in enumerate(plans[row['id']]):
                    cond=p['condition']
                    if cond=='untouched_corrupted':continue
                    if cond=='fixed_source':d=cache[p['source_id']]*np.float32(p['strength'])
                    elif cond=='recipient_self':d=baseline['recipients_clean'][index]-baseline['recipients_corrupt'][index]
                    elif cond=='zero_write':d=np.zeros((12,1024),np.float32)
                    elif cond=='norm_matched_random':d=random[p['random_draw_id']]
                    else:raise ValueError(cond)
                    assert c.array_hash(d)==p['signal_hash'];active.append(i);deltas.append(d)
                new=engine.measure([row]*len(active),deltas=np.stack(deltas));total+=len(active)
                with np.load(run/f'recipient-{index:04d}.npz') as z:old={k:z[k][active] for k in z.files}
                compare(row['id'],new,old)
            c.write(output/f'{bank}.json',[v for v in checks if v['bank']==bank]);print(bank,'original-shape replay finished',flush=True)
        errors={k:max(v['coordinate_errors'][k] for v in checks) for k in ['logits','states','source']}
        exact=all(v['old_hashes']==v['new_hashes'] for v in checks)
        passed=all(v<=cfg['numerical_absolute_tolerance'] for v in errors.values()) and sum(v['top1_changes'] for v in checks)==0
        c.write(output/'summary.json',dict(status='passed' if passed else 'failed',created_utc=c.now(),model_examples=total,
            bitwise_identical=exact,maximum_coordinate_errors=errors,checks=len(checks),top1_changes=sum(v['top1_changes'] for v in checks),
            scope='Post-audit investigation; original singleton tolerance failure is retained',environment=engine.environment))
        assert passed
    finally:engine.close()


if __name__=='__main__':main()
