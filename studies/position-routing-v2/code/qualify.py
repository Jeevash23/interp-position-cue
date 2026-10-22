"""Qualify the causal path adapter with a constant batch shape and equivalent interventions."""
import resource
import time
import numpy as np
import core as c
from mechanism import MechanismEngine,signal_bank


def main():
    output=c.ROOT/'runs/qualification';output.mkdir(exist_ok=False);cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/inputs.json')
    rows=[r for r in bundle['recipients'] if r['split']=='development' and r['name_index']==0 and r['context_index']==0]
    sources=[r for r in bundle['sources'] if r['split']=='development'];engine=MechanismEngine(cfg);start=time.perf_counter();records=[]
    try:
        sc=engine.measure(sources,True);sb=engine.measure(sources,False);raw,normalized=signal_bank(rows,sources,sc,sb)
        np.savez_compressed(output/'sources-clean.npz',**sc);np.savez_compressed(output/'sources-corrupt.npz',**sb);c.write(output/'sources.json',sources)
        with np.load(c.ROOT/'data/query-rotations.npz') as z:rot=z['rotations']
        rotation_error=float(np.max(np.abs(rot@rot.transpose(0,2,1)-np.eye(64))))
        assert rotation_error<1e-6
        for i,row in enumerate(rows):
            base=engine.forward([row]*8);wrapped,cache=engine.probe(row,np.zeros((8,12,1024),np.float32),capture_baseline=True)
            repeat,_=engine.probe(row,np.zeros((8,12,1024),np.float32))
            qbanks={'zero':engine.torch.as_tensor(wrapped['q'][0],device=engine.device),'rotations':engine.torch.as_tensor(rot,device=engine.device)}
            modes=['none','q','k','v','p','z','pv','query_only']
            specs=[dict(mode=mode,heads=cfg['consumer_heads'],bank='zero') for mode in modes]
            clamped,_=engine.probe(row,np.zeros((8,12,1024),np.float32),specs,cache,qbanks)
            errors={}
            for label,out in [('wrapped_zero',wrapped),('repeat',repeat),('baseline_path_clamps',clamped)]:
                errors[label]={k:float(np.max(np.abs(out[k]-base[k]))) for k in ['logits','states','source']}
                errors[label]['top1_changes']=int(np.sum(out['logits'].argmax(1)!=base['logits'].argmax(1)))
            delta=normalized[row['source_ids']['same_matched']]
            natural,_=engine.probe(row,np.stack([np.zeros_like(delta),delta]),capture_baseline=False)
            qbanks['signal']=engine.torch.as_tensor(natural['q'][1],device=engine.device)
            probes=[dict(mode='pv',heads=cfg['consumer_heads']),dict(mode='z',heads=cfg['consumer_heads'])]
            joint,_=engine.probe(row,np.stack([delta,delta]),probes,cache,qbanks)
            errors['pv_vs_z']={k:float(np.max(np.abs(joint[k][0]-joint[k][1]))) for k in ['logits','states','source']}
            errors['pv_vs_z']['top1_changes']=int(joint['logits'][0].argmax()!=joint['logits'][1].argmax())
            edits,_=engine.probe(row,np.zeros((2,12,1024),np.float32),[dict(mode='query_only',heads=cfg['consumer_heads'],bank='signal'),dict(mode='rotated_query',heads=cfg['consumer_heads'],bank='signal')],cache,qbanks)
            norms=[]
            for li,h in [(cfg['capture_layers'].index(l),h) for l,h in cfg['consumer_heads']]:
                a=edits['q'][0,li,h]-wrapped['q'][0,li,h];b=edits['q'][1,li,h]-wrapped['q'][0,li,h]
                norms.append(abs(float(np.linalg.norm(a)-np.linalg.norm(b)))/max(float(np.linalg.norm(a)),1e-10))
            assert max(norms)<1e-5
            passed=all(v<=cfg['numerical_absolute_tolerance'] for e in errors.values() for v in e.values())
            records.append(dict(row_id=row['id'],errors=errors,rotation_norm_relative_error=max(norms),passed=passed))
            np.savez_compressed(output/f'row-{i:02d}.npz',base_logits=base['logits'],wrapped_logits=wrapped['logits'],clamped_logits=clamped['logits'],
                base_states=base['states'],wrapped_states=wrapped['states'],clamped_states=clamped['states'],pv_z_logits=joint['logits'],pv_z_states=joint['states'],queries=edits['q'])
        allpass=all(r['passed'] for r in records)
        c.write(output/'summary.json',dict(status='passed' if allpass else 'failed',records=records,rotation_orthogonality_error=rotation_error,
            model_examples=engine.model_examples,seconds=time.perf_counter()-start,environment=engine.environment,
            peak_gpu_bytes=engine.torch.cuda.max_memory_allocated(),maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_utc=c.now()))
        print('Qualification',allpass,'on',len(rows),'prompts;',engine.model_examples,'model examples',flush=True)
        assert allpass
    finally:engine.close()


if __name__=='__main__':main()
