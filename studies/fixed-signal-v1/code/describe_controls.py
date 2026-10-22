"""Secondary source/order/dose descriptions; never fit or select the primary model."""
from collections import defaultdict
import numpy as np
import core as c
from response_models import crossed_interval


def summarize(values):
    return dict(n=len(values),**{k:float(np.mean([r[k] for r in values])) for k in
        ['correct','margin_repair','io_probability','entropy','kl_from_corrupted']},
        harmful_margin_fraction=float(np.mean([r['margin_repair']<0 for r in values])))


def main():
    output=c.ROOT/'analysis/control-descriptions.json';assert not output.exists();results={}
    for bank in ['development','holdout','confirmation']:
        folder=c.ROOT/'analysis'/bank;beh=c.read(folder/'behavior.json');pairs=c.read(folder/'pairs.json')
        groups=defaultdict(list)
        for r in beh:
            label=f"{r['condition']}|{r['cell']}|{r['source_variant']}|{r['strength']}"
            groups[label].append(r)
        slices={k:summarize(v) for k,v in groups.items()}
        strengths={str(t):summarize([r for r in beh if r['condition']=='fixed_source' and r['strength']==t]) for t in [-.5,.5,1.]}
        order={}
        for matched in [True,False]:
            values=[r for r in beh if r['condition']=='fixed_source' and r['strength']>0 and
                ((r['cell'].endswith('order0'))==(r['source_variant']=='canonical'))==matched]
            order['matching_mention_order' if matched else 'opposite_mention_order']=summarize(values)
        positive=[r for r in beh if r['condition']=='fixed_source' and r['strength']>0]
        # Aggregate all four positive-source conditions within each recipient before contrasts.
        recipients=defaultdict(dict)
        for r in beh:recipients[r['row_id']].setdefault(r['condition'],[]).append(r)
        vsrandom=[];names=[];contexts=[];vsbaseline=[]
        for conditions in recipients.values():
            pp=[r['margin_repair'] for r in conditions['fixed_source'] if r['strength']>0]
            rr=[r['margin_repair'] for r in conditions['norm_matched_random']]
            vsbaseline.append(np.mean(pp));vsrandom.append(np.mean(pp)-np.mean(rr))
            first=conditions['zero_write'][0];names.append(first['name']);contexts.append(first['context'])
        names=np.array(names);contexts=np.array(contexts);one=np.ones(len(names))
        behavior_intervals=dict(positive_repair=crossed_interval(np.array(vsbaseline),one,names,contexts),
            positive_minus_random=crossed_interval(np.array(vsrandom),one,names,contexts))
        with np.load(folder/'prediction-errors.npz') as z:errors={k:z[k] for k in z.files}
        error_slices={}
        for field in ['cell','source_variant','strength']:
            for value in sorted({p[field] for p in pairs},key=str):
                mask=np.array([p[field]==value for p in pairs]);den=errors['energy'][mask].sum()
                error_slices[f'{field}:{value}']=dict(n=int(mask.sum()),gain_error=float(errors['gain'][mask].sum()/den),
                    transform_error=float(errors['transform'][mask].sum()/den),improvement=float((errors['gain'][mask]-errors['transform'][mask]).sum()/den))
        # A diagnostic for local linearity in dose, not a new predictive endpoint.
        run=c.ROOT/'runs'/bank;rows=c.read(run/'recipients.json');plans=c.read(run/'conditions.json');planmap=defaultdict(list)
        for p in plans:planmap[p['recipient_id']].append(p)
        dose={label:dict(num=0.,den=0.,n=0,cosines=[]) for label in ['one_vs_twice_half','negative_half_vs_minus_half_one']}
        for i,row in enumerate(rows):
            pp=planmap[row['id']]
            with np.load(run/f'recipient-{i:04d}.npz') as z:states=z['states'][:,2].astype(float)
            zero=next(j for j,p in enumerate(pp) if p['condition']=='zero_write');states-=states[zero].copy()
            for source in {p['source_id'] for p in pp if p['condition']=='fixed_source'}:
                dd={p['strength']:states[j] for j,p in enumerate(pp) if p['condition']=='fixed_source' and p['source_id']==source}
                for label,pred,target in [('one_vs_twice_half',2*dd[.5],dd[1.]),('negative_half_vs_minus_half_one',-.5*dd[1.],dd[-.5])]:
                    d=dose[label];d['num']+=float(np.sum((target-pred)**2));d['den']+=float(np.sum(target**2));d['n']+=1
                    d['cosines'].append(float(np.dot(target,pred)/(np.linalg.norm(target)*np.linalg.norm(pred))))
        dose_summary={k:dict(n=d['n'],normalized_error=d['num']/d['den'],cosine_mean=float(np.mean(d['cosines']))) for k,d in dose.items()}
        results[bank]=dict(slices=slices,strengths=strengths,positive_mention_order=order,positive_strength=summarize(positive),
            behavior_intervals=behavior_intervals,prediction_slices=error_slices,dose_linearity_diagnostic=dose_summary)
    c.write(output,dict(status='complete',scope='Secondary descriptive analysis; no multiplicity-adjusted slice claims; no tuning or exclusion',
        source_sha256=c.sha(__file__),banks=results))
    print('Secondary descriptions complete',flush=True)


if __name__=='__main__':main()
