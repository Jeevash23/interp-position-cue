"""Independent score, fixed-signal, trace and crossed-inference reconstruction."""
import numpy as np
from scipy.special import logsumexp
import core as c


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def close(a,b,tol=1e-9):
    assert np.allclose(a,b,rtol=1e-10,atol=tol),float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def audit_bank(phase):
    run=c.ROOT/'runs'/phase;rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');plan=c.read(run/'conditions.json');status=c.read(run/'status.json')
    cfg=c.read(c.ROOT/'protocol/config.json');sc=arrays(run/'sources-clean.npz');sb=arrays(run/'sources-corrupt.npz');signal=arrays(run/'signals.npz')
    assert np.array_equal(sc['source']-sb['source'],signal['raw']);lookup={r['id']:i for i,r in enumerate(sources)}
    signalhash=c.read(run/'signal-hashes.json');margins=[];traces=0;conditions=0;reuse={}
    labels=['zero']+[f'{align}@{dose:g}' for dose in cfg['strengths'] for align in ['matched','opposite']]
    for ri,row in enumerate(rows):
        z=arrays(run/f'row-{ri:04d}.npz');assert z['target_logits'].shape==(len(plan),2)
        assert all(np.isfinite(v).all() for v in z.values() if v.dtype.kind not in 'US')
        assert z['write_error'].max()==0 and np.array_equal(z['target_logits'][0],z['target_logits'][1]) and np.array_equal(z['state'][0],z['state'][1])
        margin=z['target_logits'].astype(float)[:,0]-z['target_logits'].astype(float)[:,1];margins.append(margin)
        reference=signal['raw'][lookup[row['norm_reference_id']]]
        for align,sid in row['source_ids'].items():
            raw=signal['raw'][lookup[sid]];expected=raw*(np.linalg.norm(reference,axis=1,keepdims=True)/np.linalg.norm(raw,axis=1,keepdims=True))
            assert np.array_equal(expected,signal['normalized'][lookup[sid]])
            assert c.array_hash(expected)==signalhash[sid];reuse.setdefault(sid,set()).add(c.array_hash(expected))
        for i,p in enumerate(plan):
            delta=np.zeros((12,1024),np.float32) if p['kind'] in ['zero','restore'] else signal['normalized'][lookup[row['source_ids'][p['align']]]]*p['dose']
            assert c.array_hash(delta)==z['signal_hashes'][i]
            if p.get('mask')==-2:
                ni=0 if p['kind']=='clamp' else labels.index(f'{p["align"]}@{p["dose"]:g}')
                assert np.array_equal(z['target_logits'][i],z['natural_logits'][ni,[row['io_id'],row['subject_id']]])
                assert np.array_equal(z['state'][i],z['natural_state'][ni])
        assert np.array_equal(z['target_logits'][0],z['natural_logits'][0,[row['io_id'],row['subject_id']]])
        assert ('logits' in z)==(row['id'] in status['full_trace_rows'])
        if 'logits' in z:
            traces+=1;logits=z['logits'].astype(float);lz=logsumexp(logits,axis=1);lp=logits-lz[:,None]
            assert np.array_equal(z['target_logits'],z['logits'][:,[row['io_id'],row['subject_id']]])
            assert np.array_equal(z['top1'],logits.argmax(1));close(z['log_normalizer'],lz)
            close(z['entropy'],-np.sum(np.exp(lp)*lp,axis=1));close(z['kl_from_zero'],np.sum(np.exp(lp[0])[None]*(lp[0][None]-lp),axis=1))
            for i,p in enumerate(plan):
                ni=labels.index(p.get('bank','zero'))
                for key,field in [('q','q'),('attn','z')]:
                    for layer,head in p.get(key,[]):
                        li=cfg['capture_layers'].index(layer)
                        assert np.array_equal(z[f'trace_{field}'][i,li,head],z[f'natural_{field}'][ni,li,head]),(phase,ri,i,key)
                for layer in p.get('mlp',[]):
                    li=cfg['capture_layers'].index(layer)
                    assert np.array_equal(z['trace_m'][i,li],z['natural_m'][ni,li]),(phase,ri,i,'mlp')
        conditions+=len(plan)
        if (ri+1)%128==0:print(phase,'raw audit',ri+1,'/',len(rows),flush=True)
    assert all(len(v)==1 for v in reuse.values());margins=np.array(margins)
    if phase!='screen':
        saved=arrays(c.ROOT/f'analysis/{phase}/paired-scores.npz');assert np.array_equal(margins,saved['margins'])
        summary=c.read(c.ROOT/f'analysis/{phase}/summary.json');ix={p['id']:i for i,p in enumerate(plan)}
        for corruption in ['third_name','repeat_io']:
            ids=[i for i,r in enumerate(rows) if r['corruption']==corruption];rr=[rows[i] for i in ids];mm=margins[ids]
            ns=len({r['name_index'] for r in rr});cs=len({r['context_index'] for r in rr})
            for dose in cfg['strengths']:
                key=f'{corruption}@{dose:g}';out=summary['strata'][key]
                def difference(kind,mask):
                    return mm[:,ix[f'{kind}:{mask}:matched@{dose:g}']]-mm[:,ix[f'{kind}:{mask}:opposite@{dose:g}']]
                d={f'clamp_{m}':difference('clamp',m) for m in range(8)};d.update({f'restore_{m}':difference('restore',m) for m in range(1,8)})
                d['joint_necessity']=d['clamp_0']-d['clamp_7'];d['joint_sufficiency']=d['restore_7']
                d['extra_necessity']=d['clamp_1']-d['clamp_7'];d['extra_sufficiency']=d['restore_7']-d['restore_1']
                for kind in ['clamp','restore']:
                    for ctrl in ['sham_attention','sham_mlp','all']:d[f'{kind}_{ctrl}']=difference(kind,ctrl)
                for bit,name in [(2,'attention'),(4,'mlp')]:
                    d[f'{name}_specific_necessity']=d[f'clamp_sham_{name}']-d[f'clamp_{1|bit}']
                    d[f'{name}_specific_sufficiency']=d[f'restore_{1|bit}']-d[f'restore_sham_{name}']
                for a,b in [(1,2),(1,4),(2,4)]:
                    d[f'restore_interaction_{a}_{b}']=d[f'restore_{a|b}']-d[f'restore_{a}']-d[f'restore_{b}']
                    d[f'clamp_interaction_{a}_{b}']=d[f'clamp_{a}']+d[f'clamp_{b}']-d[f'clamp_{a|b}']-d['clamp_0']
                d['restore_interaction_total']=d['restore_7']-d['restore_1']-d['restore_2']-d['restore_4']
                counts=np.zeros((ns,cs));grids={k:counts.copy() for k in d}
                for i,r in enumerate(rr):
                    n,ctx=r['name_index'],r['context_index'];counts[n,ctx]+=1
                    for k,v in d.items():grids[k][n,ctx]+=v[i]
                assert np.all(counts==8)
                for k in grids:grids[k]/=counts;close(grids[k],saved[f'{key}:{k}'])
                rng=np.random.default_rng(cfg['bootstrap_seed']);ni=rng.integers(ns,size=(cfg['bootstrap_draws'],ns));ci=rng.integers(cs,size=(cfg['bootstrap_draws'],cs))
                nw=np.stack([(ni==n).sum(1) for n in range(ns)],1);cw=np.stack([(ci==n).sum(1) for n in range(cs)],1)
                boot={k:np.einsum('bi,ij,bj->b',nw,g,cw)/(ns*cs) for k,g in grids.items()}
                for k,r in out['ratios'].items():
                    close(r['estimate'],d[k].mean()/d['clamp_0'].mean());alpha=1-r['interval_level']
                    close(r['interval'],np.quantile(boot[k]/boot['clamp_0'],[alpha/2,1-alpha/2]))
                    assert r['invalid_denominators']==int(np.sum(boot['clamp_0']<=.25))
    for rel,digest in c.read(run/'manifest.json')['source_hashes'].items():assert c.sha(run/'snapshot'/rel)==digest==c.sha(c.ROOT/rel)
    return dict(status='passed',recipient_evaluations=len(rows),primary_conditions_audited=conditions,full_vocabulary_and_intervention_trace_cases=traces,
                full_trace_conditions=traces*len(plan),source_reuse_verified=True,crossed_inference_independently_reconstructed=phase!='screen')


def audit_selection():
    run=c.ROOT/'runs/screen';rows=c.read(run/'recipients.json');plan=c.read(run/'conditions.json');ix={p['id']:i for i,p in enumerate(plan)}
    mm=[]
    for i in range(len(rows)):
        z=arrays(run/f'row-{i:04d}.npz')['target_logits'].astype(float);mm.append(z[:,0]-z[:,1])
    mm=np.array(mm);q=mm[:,ix['Q:matched']]-mm[:,ix['Q:opposite']];rank=[]
    for cand in sorted({p['candidate'] for p in plan if 'candidate' in p}):
        v=q-(mm[:,ix[f'{cand}:matched']]-mm[:,ix[f'{cand}:opposite']]);rank.append((float(v.mean()),cand))
    rank.sort(key=lambda x:(-x[0],x[1]));saved=c.read(c.ROOT/'analysis/SCREEN-RANKING.json')
    assert [x[1] for x in rank]==[r['id'] for r in saved]
    for x,r in zip(rank,saved):close(x[0],r['incremental_removal'])
    s=c.read(c.ROOT/'analysis/SELECTION.json')
    assert s['attention_heads']==[r['attn'][0] for r in saved if r['attn']][:2]
    assert s['mlp_blocks']==[r['mlp'][0] for r in saved if r['mlp']][:1]


def main():
    output=c.ROOT/'analysis/EVIDENCE-AUDIT.json';assert not output.exists()
    banks={phase:audit_bank(phase) for phase in ['screen','development','confirmation']};audit_selection()
    for freeze in (c.ROOT/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze)['files'].items():assert c.sha(c.ROOT/rel)==digest,(str(freeze),rel)
    c.write(output,dict(status='passed',created_utc=c.now(),banks=banks,selection_independently_reconstructed=True,frozen_scientific_files_unchanged=True,
        source_sha256=c.sha(__file__),scope='All primary target logits and crossed effects; full-vocabulary metrics and actual site replacements independently checked on the predeclared full-trace subset.'))
    print('Independent evidence audit passed',flush=True)


if __name__=='__main__':main()
