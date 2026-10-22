"""Replay every prespecified full-trace case and all unique source pairs at original batch shape."""
import time
import numpy as np
import core as c
from path_engine import PathEngine,source_bank
from run_study import measure_row


def main():
    output=c.ROOT/'runs/replay';output.mkdir(exist_ok=False);cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/inputs.json')
    engine=PathEngine(cfg);start=time.perf_counter();checks=[];cache={};cases=0;conditions=0;sourcepairs=0
    def equal(label,new,old):
        assert new.keys()==old.keys(),label
        errors={};hashes={}
        for k,a in new.items():
            b=old[k];same=np.array_equal(a,b)
            err=0. if a.dtype.kind in 'US' and same else (None if a.dtype.kind in 'US' else float(np.max(np.abs(a.astype(float)-b.astype(float)))))
            hashes[k]=dict(old=c.array_hash(b),new=c.array_hash(a));errors[k]=err
            assert same,(label,k,err)
        checks.append(dict(label=label,bitwise_equal=True,errors=errors,hashes=hashes))
    def readarrays(path):
        with np.load(path) as z:return {k:z[k] for k in z.files}
    try:
        for phase in ['screen','development','confirmation']:
            run=c.ROOT/'runs'/phase;rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');plan=c.read(run/'conditions.json');status=c.read(run/'status.json')
            split='development' if phase=='screen' else phase
            if split not in cache:
                sc=engine.measure(sources,True);sb=engine.measure(sources,False);sourcepairs+=len(sources)
                cache[split]=(sc,sb)
            sc,sb=cache[split]
            equal(f'{phase}:sources-clean',sc,readarrays(run/'sources-clean.npz'));equal(f'{phase}:sources-corrupt',sb,readarrays(run/'sources-corrupt.npz'))
            allrows=[r for r in bundle['recipients'] if r['split']==split];_,signals=source_bank(allrows,sources,sc,sb)
            for i,row in enumerate(rows):
                if row['id'] not in status['full_trace_rows']:continue
                new,_=measure_row(engine,row,signals,plan,True);equal(f'{phase}:{row["id"]}',new,readarrays(run/f'row-{i:04d}.npz'))
                cases+=1;conditions+=len(plan)
            print(phase,'fixed-sample replay exact',flush=True)
        c.write(output/'summary.json',dict(status='passed',bitwise_identical=True,checks=checks,recipient_evaluations=cases,condition_records=conditions,
            unique_source_pairs=sourcepairs,model_examples=engine.model_examples,seconds=time.perf_counter()-start,environment=engine.environment,
            finished_utc=c.now(),maximum_coordinate_error=0.,source_sha256=c.sha(__file__)))
    finally:engine.close()


if __name__=='__main__':main()
