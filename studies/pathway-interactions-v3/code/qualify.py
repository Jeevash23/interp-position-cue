"""Qualify no-op, source-boundary and complete END-path interventions before selection."""
import time,resource
import numpy as np
import core as c
from path_engine import PathEngine,source_bank


def main():
    cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/inputs.json')
    rows=[r for r in bundle['recipients'] if r['split']=='development'];sources=[r for r in bundle['sources'] if r['split']=='development']
    sample=[r for r in rows if r['name_index']==0 and r['context_index']==0 and r['role']==0]
    output=c.ROOT/'runs/qualification';output.mkdir(exist_ok=False);engine=PathEngine(cfg);start=time.perf_counter();records=[]
    try:
        sc=engine.measure(sources,True);sb=engine.measure(sources,False);raw,signals=source_bank(rows,sources,sc,sb)
        for label,values in [('clean',sc),('corrupt',sb)]:np.savez_compressed(output/f'sources-{label}.npz',**values)
        c.write(output/'sources.json',sources)
        allsites=dict(q=[],attn=[[l,h] for l in cfg['capture_layers'] for h in range(16)],mlp=cfg['capture_layers'])
        partial=dict(q=cfg['consumer_heads'],attn=[[7,0],[23,0]],mlp=[12])
        for i,row in enumerate(sample):
            zero=np.zeros((12,1024),np.float32);d=signals[row['source_ids']['matched']]
            original=engine.forward([row]*8)
            natural=engine.probe(row,np.stack([zero,d]));banks=engine.banks(natural,['zero','signal'])
            specs=[{},dict(bank='zero',**partial),dict(bank='zero',**allsites),dict(bank='zero',**allsites),dict(bank='signal',**allsites),dict(bank='signal',**partial)]
            out=engine.probe(row,np.stack([zero,zero,zero,d,zero,d]),specs,banks)
            errors={};top1_changes={}
            for label,a,b in [('wrapped',natural['logits'][0],original['logits'][0]),('wrapped_state',natural['state'][0],original['states'][0,-1]),
                 ('repeat',out['logits'][0],natural['logits'][0]),('partial_noop',out['logits'][1],natural['logits'][0]),
                 ('all_noop',out['logits'][2],natural['logits'][0]),('remove_all',out['logits'][3],natural['logits'][0]),
                 ('restore_all',out['logits'][4],natural['logits'][1]),('signal_noop',out['logits'][5],natural['logits'][1]),
                 ('remove_all_state',out['state'][3],natural['state'][0]),('restore_all_state',out['state'][4],natural['state'][1])]:
                errors[label]=float(np.max(np.abs(a-b)))
                if a.shape[-1]==50257:top1_changes[label]=int(a.argmax()!=b.argmax())
            assert max(errors.values())<=cfg['numerical_absolute_tolerance'] and out['write_error'].max()==0 and max(top1_changes.values())==0
            records.append(dict(row_id=row['id'],errors=errors,top1_changes=top1_changes,passed=True))
            np.savez_compressed(output/f'row-{i:02d}.npz',original_logits=original['logits'],original_state=original['states'][:,-1],
                                natural_logits=natural['logits'],natural_state=natural['state'],logits=out['logits'],state=out['state'])
        c.write(output/'summary.json',dict(status='passed',records=records,model_examples=engine.model_examples,seconds=time.perf_counter()-start,
            maximum_error=max(max(r['errors'].values()) for r in records),environment=engine.environment,peak_gpu_bytes=engine.torch.cuda.max_memory_allocated(),
            maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=c.now()))
        print('Qualification passed:',len(sample),'cases;',engine.model_examples,'model examples',flush=True)
    finally:engine.close()


if __name__=='__main__':main()
