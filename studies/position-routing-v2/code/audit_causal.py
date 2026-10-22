"""Independent raw-logit, path-intervention, fixed-signal and crossed-bootstrap audit."""
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp,softmax
import core as c


def close(a,b,tol=1e-9):
    assert np.allclose(a,b,rtol=1e-10,atol=tol),float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def arrays(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def audit_bank(bank):
    run=c.ROOT/'runs'/bank;analysis=c.ROOT/'analysis'/bank;cfg=c.read(c.ROOT/'protocol/config.json');bundle=c.read(c.ROOT/'data/inputs.json')
    rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');plans=c.read(run/'conditions.json')
    assert len({(p['row_id'],p['id']) for p in plans})==len(plans)==45*len(rows)
    pp=defaultdict(list);bb=defaultdict(list)
    for p in plans:pp[p['row_id']].append(p)
    for b in c.read(analysis/'behavior.json'):bb[b['row_id']].append(b)
    sc=arrays(run/'sources-clean.npz');sb=arrays(run/'sources-corrupt.npz');signals=arrays(run/'signals.npz');hashes=c.read(run/'signal-hashes.json')
    assert np.array_equal(sc['source']-sb['source'],signals['raw'])
    cache={r['id']:i for i,r in enumerate(sources)};labels=[p['id'] for p in pp[rows[0]['id']]];ix={p:i for i,p in enumerate(labels)}
    with np.load(c.ROOT/'data/query-rotations.npz') as z:rotations=z['rotations'].astype(float)
    close(rotations@rotations.transpose(0,2,1),np.repeat(np.eye(64)[None],5,axis=0),tol=1e-6)
    for sid,i in cache.items():assert c.array_hash(signals['raw'][i])==hashes[sid]['raw'] and c.array_hash(signals['normalized'][i])==hashes[sid]['normalized']
    margins=[];max_k=0.;max_v=0.;max_rotation=0.;reused=defaultdict(set)
    for ri,row in enumerate(rows):
        raw=arrays(run/f'recipient-{ri:04d}.npz');plans_row=pp[row['id']];beh=bb[row['id']]
        assert len(beh)==45 and raw['logits'].shape==(45,50257);assert all(np.isfinite(v).all() for v in raw.values())
        assert [p['id'] for p in plans_row]==labels==[b['condition'] for b in beh]
        for key in ['logits','states','source','q','p','z']:assert np.array_equal(raw[key][0],raw[key][ix['zero_write']])
        logits=raw['logits'].astype(float);lp=logits-logsumexp(logits,axis=1)[:,None];margin=logits[:,row['io_id']]-logits[:,row['subject_id']];margins.append(margin)
        close(raw['p'].sum(-1),1,tol=1e-6);assert raw['p'].min()>=0
        for ci,(plan,b) in enumerate(zip(plans_row,beh)):
            assert plan['status']=='complete';mode=plan['mode']
            if plan['scale']=='zero':delta=np.zeros((12,1024),np.float32)
            elif plan['scale']=='self':delta=raw['clean_source'][0]-raw['source'][0]
            else:
                sid=row['source_ids'][plan['source']];assert sid==plan['source_id']
                delta=signals[plan['scale']][cache[sid]];reused[(sid,plan['scale'])].add(plan['signal_hash'])
                ref=signals['raw'][cache[row['norm_reference_id']]]
                if plan['scale']=='normalized':
                    original=signals['raw'][cache[sid]];expected=original*(np.linalg.norm(ref,axis=1,keepdims=True)/np.linalg.norm(original,axis=1,keepdims=True))
                    assert np.array_equal(delta,expected)
            assert c.array_hash(delta)==plan['signal_hash']
            assert b['top1']==int(logits[ci].argmax()) and b['correct']==(logits[ci].argmax()==row['io_id'])
            close(b['margin'],margin[ci]);close(b['margin_repair'],margin[ci]-margin[0])
            close(b['entropy'],-np.sum(np.exp(lp[ci])*lp[ci]));close(b['io_probability'],np.exp(lp[ci,row['io_id']]))
            close(b['kl_from_zero'],np.sum(np.exp(lp[0])*(lp[0]-lp[ci])))
            for layer,h in plan.get('heads',[]):
                li=cfg['capture_layers'].index(layer)
                if mode=='q':assert np.array_equal(raw['q'][ci,li,h],raw['q'][0,li,h])
                elif mode=='p':assert np.array_equal(raw['p'][ci,li,h],raw['p'][0,li,h])
                elif mode=='z':assert np.array_equal(raw['z'][ci,li,h],raw['z'][0,li,h])
                elif mode=='k':
                    predicted=softmax(raw['q'][ci,li,h].astype(float)@raw['baseline_k'][li,h].astype(float).T/8)
                    error=float(np.max(np.abs(predicted-raw['p'][ci,li,h])));max_k=max(max_k,error);assert error<1e-5
                elif mode=='v':
                    predicted=raw['p'][ci,li,h].astype(float)@raw['baseline_v'][li,h].astype(float)
                    error=float(np.max(np.abs(predicted-raw['z'][ci,li,h])));max_v=max(max_v,error);assert error<1e-5
                elif mode in ['query_only','rotated_query']:
                    bi=ix[plan['bank']];assert c.array_hash(raw['q'][bi])==plan['query_bank_hash']
                    baseq=raw['q'][0,li,h].astype(float);change=raw['q'][bi,li,h].astype(float)-baseq
                    if mode=='query_only':assert np.array_equal(raw['q'][ci,li,h],raw['q'][bi,li,h])
                    else:
                        hi=cfg['consumer_heads'].index([layer,h]);expected=baseq+change@rotations[hi]
                        close(raw['q'][ci,li,h],expected,tol=1e-5)
                        relative=abs(np.linalg.norm(raw['q'][ci,li,h].astype(float)-baseq)/np.linalg.norm(change)-1)
                        max_rotation=max(max_rotation,float(relative));assert relative<1e-5
        if (ri+1)%64==0:print(bank,'audited',ri+1,'/',len(rows),flush=True)
    assert all(len(v)==1 for v in reused.values());margins=np.array(margins)
    paired=arrays(analysis/'paired-scores.npz');assert np.array_equal(margins,paired['margins'])
    summary=c.read(analysis/'summary.json');series={}
    def difference(kind,suffix=''):return margins[:,ix[f'{kind}_matched{suffix}']]-margins[:,ix[f'{kind}_opposite{suffix}']]
    series['same']=difference('same');series['foreign']=difference('foreign');series['q_removed']=series['same']-difference('same',':clamp_q')
    series['same_query_only']=difference('same',':query_only')
    series['q_specific_vs_controls']=(difference('same',':control0_q')+difference('same',':control1_q'))/2-difference('same',':clamp_q')
    series['q_specific_vs_values']=difference('same',':clamp_v')-difference('same',':clamp_q')
    series['q_sufficiency_vs_rotation']=difference('same',':query_only')-difference('same',':rotated_query')
    ns=len({r['name_index'] for r in rows});cs=len({r['context_index'] for r in rows});grids={k:np.zeros((ns,cs)) for k in series};counts=np.zeros((ns,cs))
    for i,r in enumerate(rows):
        n,ctx=r['name_index'],r['context_index'];counts[n,ctx]+=1
        for k,v in series.items():grids[k][n,ctx]+=v[i]
    assert np.all(counts==8)
    for k in grids:grids[k]/=counts
    rng=np.random.default_rng(2026100814);ni=rng.integers(ns,size=(20000,ns));ci=rng.integers(cs,size=(20000,cs))
    nw=np.stack([(ni==n).sum(1) for n in range(ns)],axis=1);cw=np.stack([(ci==ctx).sum(1) for ctx in range(cs)],axis=1)
    boot={k:np.einsum('bi,ij,bj->b',nw,g,cw)/(ns*cs) for k,g in grids.items()}
    for name,key in dict(foreign_retention='foreign',query_necessity='q_removed',query_sufficiency='same_query_only',specificity_vs_random_heads='q_specific_vs_controls',specificity_vs_values='q_specific_vs_values',sufficiency_vs_rotation='q_sufficiency_vs_rotation').items():
        r=summary['ratios'][name];values=boot[key]/boot['same'];alpha=1-r['interval_level']
        close(r['estimate'],series[key].mean()/series['same'].mean());close(r['interval'],np.quantile(values,[alpha/2,1-alpha/2]))
        assert r['bootstrap_denominators_below_floor']==int(np.sum(boot['same']<=.25))
    for rel,digest in c.read(run/'manifest.json')['source_hashes'].items():assert c.sha(run/'snapshot'/rel)==digest==c.sha(c.ROOT/rel)
    return dict(status='passed',recipients=len(rows),conditions=len(plans),fixed_signal_groups=len(reused),
        maximum_key_clamp_probability_error=max_k,maximum_value_clamp_output_error=max_v,maximum_rotated_query_norm_relative_error=max_rotation,
        full_vocabulary_scores_reconstructed=True,path_interventions_verified=True,crossed_bootstrap_independently_reconstructed=True)


def main():
    output=c.ROOT/'analysis/EVIDENCE-AUDIT.json';assert not output.exists()
    banks={bank:audit_bank(bank) for bank in ['development','confirmation']}
    for freeze in (c.ROOT/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze)['files'].items():assert c.sha(c.ROOT/rel)==digest,(str(freeze),rel)
    previous=c.read(c.ROOT/'provenance/PREVIOUS-STUDY.json');assert c.sha(previous['path'])==previous['sha256']
    c.write(output,dict(status='passed',created_utc=c.now(),banks=banks,frozen_files_unchanged=True,previous_study_manifest_unchanged=True,source_sha256=c.sha(__file__)))
    print('Causal evidence audit passed',flush=True)


if __name__=='__main__':main()
