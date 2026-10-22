"""Fresh singleton replay of fixed corner blocks; original cached signals remain fixed."""
from collections import defaultdict
import numpy as np
import core as c


def main():
    output=c.ROOT/'runs/model-replay';output.mkdir(exist_ok=False)
    cfg=c.read(c.ROOT/'protocol/config.json');engine=c.Engine(cfg)
    c.snapshot(output,[c.ROOT/'code/replay_model.py',c.ROOT/'code/core.py',c.ROOT/'protocol/config.json'])
    results={};total=0
    try:
        for bank in ['development','holdout','confirmation']:
            run=c.ROOT/'runs'/bank;dest=output/bank;dest.mkdir()
            rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');plans=c.read(run/'conditions.json')
            nmax=max(r['name_index'] for r in rows);cmax=max(r['context_index'] for r in rows)
            ri=[i for i,r in enumerate(rows) if r['name_index'] in [0,nmax] and r['context_index'] in [0,cmax]]
            si=[i for i,r in enumerate(sources) if r['name_index'] in [0,nmax]]
            chosen=[rows[i] for i in ri];ss=[sources[i] for i in si]
            checks=[]
            def compare(label,new,old):
                errors={k:float(np.max(np.abs(new[k]-old[k]))) for k in new}
                changes=int(np.sum(new['logits'].argmax(1)!=old['logits'].argmax(1)))
                checks.append(dict(label=label,coordinate_errors=errors,top1_changes=changes))
            baseline={}
            for kind,selected,indices in [('sources',ss,si),('recipients',chosen,ri)]:
                for clean in [True,False]:
                    label=f'{kind}_{"clean" if clean else "corrupt"}';new=engine.measure(selected,clean,batch_size=1);total+=len(selected)
                    with np.load(run/f'{label}.npz') as z:old={k:z[k][indices] for k in z.files}
                    compare(label,new,old);np.savez_compressed(dest/f'{label}.npz',**new);baseline[label]=old
            with np.load(run/'source-cache.npz') as z:cache={r['id']:d for r,d in zip(sources,z['deltas'])}
            with np.load(run/'random-cache.npz') as z:random={k:z[k] for k in z.files}
            byrow=defaultdict(list)
            for p in plans:byrow[p['recipient_id']].append(p)
            for local,index in enumerate(ri):
                row=rows[index];pp=byrow[row['id']];deltas=[];active=[]
                for ci,p in enumerate(pp):
                    cond=p['condition']
                    if cond=='untouched_corrupted':continue
                    if cond=='zero_write':d=np.zeros((12,1024),np.float32)
                    elif cond=='recipient_self':d=baseline['recipients_clean']['source'][local]-baseline['recipients_corrupt']['source'][local]
                    elif cond=='fixed_source':d=cache[p['source_id']]*np.float32(p['strength'])
                    elif cond=='norm_matched_random':d=random[p['random_draw_id']]
                    else:raise ValueError(cond)
                    assert c.array_hash(d)==p['signal_hash'];deltas.append(d);active.append(ci)
                new=engine.measure([row]*len(active),deltas=np.stack(deltas),batch_size=1);total+=len(active)
                with np.load(run/f'recipient-{index:04d}.npz') as z:old={k:z[k][active] for k in z.files}
                compare(row['id'],new,old);np.savez_compressed(dest/f'recipient-{index:04d}.npz',**new)
            errors={k:max(v['coordinate_errors'][k] for v in checks) for k in ['logits','states','source']}
            top1=sum(v['top1_changes'] for v in checks)
            results[bank]=dict(recipient_indices=ri,source_indices=si,checks=checks,maximum_coordinate_errors=errors,top1_changes=top1,
                passed=all(v<=cfg['numerical_absolute_tolerance'] for v in errors.values()) and top1==0)
            c.write(dest/'summary.json',results[bank]);print(bank,errors,'top1 changes',top1,flush=True)
        passed=all(v['passed'] for v in results.values())
        c.write(output/'summary.json',dict(status='passed' if passed else 'failed',created_utc=c.now(),
            singleton_model_examples=total,banks=results,environment=engine.environment,tolerance=cfg['numerical_absolute_tolerance']))
        assert passed,'Numerical replay exceeded predeclared tolerance; retain all evidence and investigate'
    finally:engine.close()


if __name__=='__main__':main()
