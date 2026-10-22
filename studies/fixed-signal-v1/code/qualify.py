"""Development competence and implementation qualification before holdout exposure."""
import json
from pathlib import Path
import resource
import time
import numpy as np
import core as c


def cells(rows,clean,corrupt,source=False):
    b=c.behavior(clean['logits'],rows);d=c.behavior(corrupt['logits'],rows);out={}
    for family in sorted({r['family'] for r in rows}):
        for order in [0,1]:
            mask=np.array([r['family']==family and r['order']==order for r in rows]);sep=b['margin'][mask]-d['margin'][mask]
            out[f'{family}/order{order}']=dict(n=int(mask.sum()),clean_top1=float(b['correct'][mask].mean()),
                clean_margin=float(b['margin'][mask].mean()),corrupted_margin=float(d['margin'][mask].mean()),
                separation=float(sep.mean()),positive_separation=float((sep>0).mean()))
    return out


def run():
    config=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/pilot-qualified.json')
    output=c.ROOT/'runs/qualification';output.mkdir(exist_ok=False)
    files=[c.ROOT/'code/core.py',Path(__file__),c.ROOT/'code/prepare_inputs.py',c.ROOT/'protocol/config.json',c.ROOT/'protocol/QUALIFICATION.md',c.ROOT/'data/pilot-qualified.json']
    source_hashes=c.snapshot(output,files)
    c.write(output/'manifest.json',dict(created_utc=c.now(),source_hashes=source_hashes,split='development_only'))
    started=time.perf_counter();engine=c.Engine(config)
    try:
        results={};allrows={}
        for kind in ['sources','recipients']:
            rows=[r for r in bundle[kind] if r['split']=='development'];allrows[kind]=rows;results[kind]={}
            for clean in [True,False]:
                v=engine.measure(rows,clean);results[kind]['clean' if clean else 'corrupt']=v
                np.savez_compressed(output/f'{kind}-{"clean" if clean else "corrupt"}.npz',**v)
            c.write(output/f'{kind}.json',rows)
        src=results['sources']['clean']['source']-results['sources']['corrupt']['source']
        np.savez_compressed(output/'source-cache.npz',deltas=src)
        source_hashes_by_id={r['id']:c.array_hash(v) for r,v in zip(allrows['sources'],src)}
        rr=[r for r in allrows['recipients'] if len(r['tokens_clean'])==len(allrows['recipients'][0]['tokens_clean'])][:4]
        base=engine.forward(rr);zero=engine.forward(rr,deltas=np.zeros((len(rr),12,1024),np.float32));repeat=engine.forward(rr)
        singleton=engine.measure(rr,batch_size=1)
        errors={}
        for label,value in [('zero_write',zero),('repeat',repeat),('singleton',singleton)]:
            errors[label]={key:float(np.max(np.abs(value[key]-base[key]))) for key in ['logits','states','source']}
            errors[label]['top1_changes']=int(np.sum(value['logits'].argmax(1)!=base['logits'].argmax(1)))
        layer,head=engine.source[0];rng=np.random.default_rng(config['seed'])
        dz=rng.normal(0,.01,size=(len(rr),64)).astype(np.float32)
        w=engine.model.transformer.h[layer].attn.c_proj.weight[head*64:(head+1)*64].detach().cpu().numpy()
        delta=np.zeros((len(rr),12,1024),np.float32);delta[:,0]=dz@w
        before=engine.forward(rr,individual_delta=(layer,head,dz));after=engine.forward(rr,deltas=delta)
        errors['projection_route']={key:float(np.max(np.abs(before[key]-after[key]))) for key in ['logits','states']}
        errors['projection_route']['top1_changes']=int(np.sum(before['logits'].argmax(1)!=after['logits'].argmax(1)))
        selfdelta=results['recipients']['clean']['source']-results['recipients']['corrupt']['source']
        selfresult=engine.measure(allrows['recipients'],deltas=selfdelta)
        np.savez_compressed(output/'recipients-self.npz',**selfresult)
        qualification={kind:cells(allrows[kind],results[kind]['clean'],results[kind]['corrupt']) for kind in allrows}
        numeric=all(value<=config['numerical_absolute_tolerance'] for record in errors.values() for key,value in record.items())
        competent=all(v['clean_top1']>=config['clean_top1_minimum'] and v['separation']>=config['clean_corrupt_margin_separation_minimum'] and
                      v['positive_separation']>=config['positive_margin_separation_fraction_minimum'] for record in qualification.values() for v in record.values())
        b=c.behavior(selfresult['logits'],allrows['recipients']);bad=c.behavior(results['recipients']['corrupt']['logits'],allrows['recipients'])
        summary=dict(status='complete',numerical_passed=numeric,competence_passed=competent,cells=qualification,numerical_errors=errors,
            source_hashes=source_hashes_by_id,self_correction=dict(top1=float(b['correct'].mean()),mean_margin_repair=float((b['margin']-bad['margin']).mean())),
            environment=engine.environment,seconds=time.perf_counter()-started,maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            gpu_peak_allocated_bytes=engine.torch.cuda.max_memory_allocated(),completed_utc=c.now())
        c.write(output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
        assert numeric,'Numerical qualification failed'
    finally:engine.close()


if __name__=='__main__':run()
