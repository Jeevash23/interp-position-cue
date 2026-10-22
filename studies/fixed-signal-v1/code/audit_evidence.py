"""Independently reconstruct recorded contrasts, scores and provenance; no model runs."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
import core as c


def load(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def close(a,b,atol=1e-9):
    assert np.allclose(a,b,atol=atol,rtol=1e-10),float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


def audit_bank(bank):
    run=c.ROOT/'runs'/bank;analysis=c.ROOT/'analysis'/bank
    rows=c.read(run/'recipients.json');sources=c.read(run/'sources.json');plans=c.read(run/'conditions.json')
    status=c.read(run/'status.json');assert status['status']=='complete' and not status['failures']
    assert len({p['id'] for p in plans})==len(plans)==13*len(rows)
    sc=load(run/'sources_clean.npz');sb=load(run/'sources_corrupt.npz')
    rc=load(run/'recipients_clean.npz');rb=load(run/'recipients_corrupt.npz')
    source=load(run/'source-cache.npz')['deltas'];random=load(run/'random-cache.npz')
    assert np.array_equal(source,sc['source']-sb['source'])
    cache={r['id']:d for r,d in zip(sources,source)};hashes=c.read(run/'source-hashes.json')
    assert hashes=={k:c.array_hash(v) for k,v in cache.items()}==status['source_cache_hashes']
    for row in rows+sources:
        a=np.array(row['tokens_clean']);b=np.array(row['tokens_corrupt'])
        assert a.shape==b.shape and np.where(a!=b)[0].tolist()==[row['s2']]
        assert a[0]==50256 and a[row['s2']]==row['subject_id'] and b[row['s2']]==row['alt_id']
        assert len({row['subject_id'],row['io_id'],row['alt_id']})==3
        assert np.sum(a==row['subject_id'])==2 and np.sum(a==row['io_id'])==1
        assert row['answer_position']==len(a)-1
    byrow=defaultdict(list)
    for p in plans:byrow[p['recipient_id']].append(p)
    beh=c.read(analysis/'behavior.json');assert len(beh)==len(plans)
    bmap=defaultdict(list)
    for b in beh:bmap[b['row_id']].append(b)
    signals=defaultdict(set);responses={};maxzero=0.;maxnorm=0.
    for ri,row in enumerate(rows):
        pp=byrow[row['id']];bb=bmap[row['id']]
        assert Counter(p['condition'] for p in pp)=={'fixed_source':6,'untouched_corrupted':1,'zero_write':1,'recipient_self':1,'norm_matched_random':4}
        raw=load(run/f'recipient-{ri:04d}.npz')
        assert raw['logits'].shape==(13,50257) and raw['states'].shape==(13,3,1024) and raw['source'].shape==(13,12,1024)
        assert all(np.isfinite(v).all() for v in raw.values())
        zero=next(i for i,p in enumerate(pp) if p['condition']=='zero_write')
        untouched=next(i for i,p in enumerate(pp) if p['condition']=='untouched_corrupted')
        for key,v in raw.items():
            assert np.array_equal(v[untouched],rb[key][ri])
            error=float(np.max(np.abs(v[zero]-rb[key][ri])));maxzero=max(maxzero,error)
            assert error<=.0005
        logits=raw['logits'].astype(float);logp=logits-logsumexp(logits,axis=1)[:,None]
        base=rb['logits'][ri].astype(float);basep=base-logsumexp(base)
        margin=logits[:,row['io_id']]-logits[:,row['subject_id']]
        base_margin=base[row['io_id']]-base[row['subject_id']]
        for j,(p,b) in enumerate(zip(pp,bb)):
            assert p['condition']==b['condition'] and p['strength']==b['strength'] and p['signal_hash']==b['signal_hash']
            cond=p['condition'];delta=None
            if cond=='fixed_source':delta=cache[p['source_id']]*np.float32(p['strength'])
            elif cond=='zero_write':delta=np.zeros((12,1024),np.float32)
            elif cond=='recipient_self':delta=rc['source'][ri]-rb['source'][ri]
            elif cond=='norm_matched_random':
                delta=random[p['random_draw_id']]
                wanted=np.linalg.norm(cache[p['source_id']],axis=1)
                got=np.linalg.norm(delta,axis=1);relative=float(np.max(np.abs(got-wanted)/wanted));maxnorm=max(maxnorm,relative)
                assert relative<1e-6
                rng=np.random.default_rng(p['random_seed']);expected=rng.normal(size=(12,1024)).astype(np.float32)
                expected*=wanted[:,None]/np.linalg.norm(expected,axis=1,keepdims=True)
                assert np.array_equal(delta,expected)
            assert p['signal_hash']==(None if delta is None else c.array_hash(delta))
            assert p['unscaled_source_hash']==hashes.get(p.get('source_id'))
            if cond in ['fixed_source','norm_matched_random']:
                signals[(cond,p['source_id'],p['strength'],p.get('random_draw_id'))].add(p['signal_hash'])
            assert b['top1']==int(logits[j].argmax()) and b['correct']==(int(logits[j].argmax())==row['io_id'])
            assert b['clean_correct']==(int(rc['logits'][ri].argmax())==row['io_id'])
            close(b['margin'],margin[j]);close(b['margin_repair'],margin[j]-base_margin)
            close(b['io_probability'],np.exp(logp[j,row['io_id']]))
            close(b['entropy'],-np.sum(np.exp(logp[j])*logp[j]))
            close(b['kl_from_corrupted'],np.sum(np.exp(basep)*(basep-logp[j])))
            state=raw['states'][j].astype(float)-raw['states'][zero].astype(float)
            close(b['response_norms'],np.linalg.norm(state,axis=1))
            if cond=='fixed_source':
                responses[(row['name_index'],row['context_index'],row['role'],b['source_variant'],p['strength'],b['cell'])]=state
        if (ri+1)%128==0:print(f'{bank}: audited {ri+1}/{len(rows)} recipients',flush=True)
    assert all(len(v)==1 for v in signals.values())
    pairs=c.read(analysis/'pairs.json');arrays=load(analysis/'response-pairs.npz');errs=load(analysis/'prediction-errors.npz')
    for i,p in enumerate(pairs):
        key=(p['name'],p['context'],p['role'],p['source_variant'],p['strength'])
        assert np.array_equal(arrays['x'][i],responses[(*key,'F10/order0')])
        assert np.array_equal(arrays['y'][i],responses[(*key,p['cell'])])
    x=arrays['x'][:,2];y=arrays['y'][:,2];model=c.read(c.ROOT/'analysis/development/model.json')
    coeff=load(c.ROOT/'analysis/development/model.npz');projection=coeff['projection']
    close(projection.T@projection,np.eye(model['rank']))
    gain=np.array([model['gains'][p['gain_key']] for p in pairs])[:,None]*x;rich=gain.copy()
    for i,p in enumerate(pairs):rich[i]+=(x[i]@projection)@coeff[f"map_{model['cells'].index(p['cell'])}"]@projection.T
    close(errs['gain_predictions'],gain);close(errs['transform_predictions'],rich)
    energy=(y*y).sum(1);eg=((y-gain)**2).sum(1);er=((y-rich)**2).sum(1)
    close(errs['energy'],energy);close(errs['gain'],eg);close(errs['transform'],er)
    pg=eg-((y-gain)*x).sum(1)**2/(x*x).sum(1);pr=er-((y-rich)*x).sum(1)**2/(x*x).sum(1)
    close(errs['perpendicular_gain'],np.maximum(pg,0));close(errs['perpendicular_transform'],np.maximum(pr,0))
    s=c.read(analysis/'summary.json');close(s['gain_normalized_error'],eg.sum()/energy.sum());close(s['transform_normalized_error'],er.sum()/energy.sum())
    names=sorted({p['name'] for p in pairs});contexts=sorted({p['context'] for p in pairs})
    for field,values in [('improvement',eg-er),('perpendicular_improvement',pg-pr)]:
        num=np.zeros((len(names),len(contexts)));den=num.copy()
        for i,p in enumerate(pairs):num[p['name'],p['context']]+=values[i];den[p['name'],p['context']]+=energy[i]
        close(s[field]['numerator_grid'],num);close(s[field]['denominator_grid'],den)
        # Independently build cluster count weights rather than reuse analysis indexing.
        rng=np.random.default_rng(2026100809);ni=rng.integers(len(names),size=(20000,len(names)));ci=rng.integers(len(contexts),size=(20000,len(contexts)))
        nw=np.array([(ni==k).sum(1) for k in names]).T;cw=np.array([(ci==k).sum(1) for k in contexts]).T
        n=np.einsum('bi,ij,bj->b',nw,num,cw);d=np.einsum('bi,ij,bj->b',nw,den,cw)
        close(s[field]['ci95'],np.quantile(n/d,[.025,.975]))
    for freeze in (c.ROOT/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze).get('files',{}).items():assert c.sha(c.ROOT/rel)==digest,(str(freeze),rel)
    manifest=c.read(run/'manifest.json')
    for rel,digest in manifest['source_hashes'].items():assert c.sha(run/'snapshot'/rel)==digest==c.sha(c.ROOT/rel)
    frozen=c.read(c.ROOT/'protocol/HELDOUT-FREEZE.json')['created_utc']
    if bank!='development':assert frozen<manifest['started_utc']
    return dict(status='passed',recipients=len(rows),sources=len(sources),conditions=len(plans),pairs=len(pairs),
        source_hashes=len(hashes),reused_signal_groups=len(signals),maximum_zero_coordinate_error=maxzero,
        maximum_random_head_norm_relative_error=maxnorm,independent_logit_scores=True,
        independent_prediction_errors=True,independent_cluster_weight_bootstrap=True,frozen_files_unchanged=True)


def main(output):
    output=Path(output);assert not output.exists()
    results={bank:audit_bank(bank) for bank in ['development','holdout','confirmation']}
    # Crossed CV must exclude both held-out axes, and every evaluation item appears once.
    pairs=c.read(c.ROOT/'analysis/development/pairs.json');selection=c.read(c.ROOT/'analysis/development/selection.json')
    seen=Counter()
    for fold in selection['folds']:
        train=[p for p in pairs if p['name']!=fold['name'] and p['context']!=fold['context']]
        test=[i for i,p in enumerate(pairs) if p['name']==fold['name'] and p['context']==fold['context']]
        assert len(train)==fold['train'] and len(test)==fold['test'];seen.update(test)
    assert len(seen)==len(pairs) and set(seen.values())=={1}
    prep=c.read(c.ROOT/'provenance/PREPARATION-SNAPSHOT.json')
    for rel,digest in prep['files'].items():assert c.sha(c.PROJECT/rel)==digest==c.sha(c.ROOT/'provenance/pre-execution'/rel)
    for item in prep['reference_files']:assert c.sha(c.FIELD/item['path_from_field_root'])==item['sha256']
    model=c.read(c.ROOT/'provenance/MODEL-ASSETS.json')
    for rel,item in model['files'].items():assert c.sha(c.MODEL/rel)==item['sha256']
    extra={k:v for k,v in prep.items() if k not in ['created_utc','files']}
    c.write(output,dict(status='passed',created_utc=c.now(),banks=results,development_folds=16,
        preparation_files_preserved=len(prep['files']),reference_files_preserved=len(prep['reference_files']),
        model_files_preserved=len(model['files']),preparation_metadata=extra,audit_source_sha256=c.sha(__file__)))
    print('Evidence audit passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
