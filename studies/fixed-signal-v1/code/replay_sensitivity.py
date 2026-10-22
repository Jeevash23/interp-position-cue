"""Post-audit effect of singleton rounding on the fixed sampled response contrasts."""
from collections import defaultdict
import numpy as np
import core as c
from analyze_responses import load_model
from response_models import predict


def main():
    output=c.ROOT/'analysis/REPLAY-SENSITIVITY.json';assert not output.exists()
    model=load_model(c.ROOT/'analysis/development');results={}
    for bank in ['development','holdout','confirmation']:
        run=c.ROOT/'runs'/bank;replay=c.ROOT/'runs/model-replay'/bank
        rows=c.read(run/'recipients.json');indices=c.read(replay/'summary.json')['recipient_indices']
        plans=defaultdict(list)
        for p in c.read(run/'conditions.json'):plans[p['recipient_id']].append(p)
        oldresponses={};newresponses={};maxrel=0.;minnorm=float('inf')
        for index in indices:
            row=rows[index];pp=plans[row['id']];active=[i for i,p in enumerate(pp) if p['condition']!='untouched_corrupted']
            with np.load(run/f'recipient-{index:04d}.npz') as z:old=z['states'][active,2].astype(float)
            with np.load(replay/f'recipient-{index:04d}.npz') as z:new=z['states'][:,2].astype(float)
            zero=next(j for j,i in enumerate(active) if pp[i]['condition']=='zero_write')
            old-=old[zero].copy();new-=new[zero].copy()
            for j,i in enumerate(active):
                p=pp[i]
                if p['condition']!='fixed_source':continue
                norm=np.linalg.norm(old[j]);maxrel=max(maxrel,float(np.linalg.norm(new[j]-old[j])/norm));minnorm=min(minnorm,float(norm))
                key=(row['name_index'],row['context_index'],row['role'],p['source_id'].rsplit(':',1)[1],p['strength'],f"{row['family']}/order{row['order']}")
                oldresponses[key]=old[j];newresponses[key]=new[j]
        statistics={}
        for label,responses in [('original_batch',oldresponses),('singleton',newresponses)]:
            keys=[key for key in responses if key[-1]!='F10/order0'];x=np.array([responses[(*k[:-1],'F10/order0')] for k in keys]);y=np.array([responses[k] for k in keys])
            gainkeys=[f'{k[-1]}|{k[3]}|{k[4]}' for k in keys];cells=[k[-1] for k in keys]
            gain=np.array([model['gains'][k] for k in gainkeys])[:,None]*x;rich=predict(model,x,gainkeys,cells)
            energy=np.sum(y*y);eg=float(np.sum((y-gain)**2)/energy);er=float(np.sum((y-rich)**2)/energy)
            statistics[label]=dict(pairs=len(keys),gain_error=eg,transform_error=er,improvement=eg-er)
        difference={k:statistics['singleton'][k]-statistics['original_batch'][k] for k in ['gain_error','transform_error','improvement']}
        results[bank]=dict(maximum_response_relative_L2_difference=maxrel,minimum_sample_response_norm=minnorm,
            statistics=statistics,changes=difference)
    replay=c.read(c.ROOT/'runs/model-replay/summary.json');maximum=max(v['maximum_coordinate_errors']['states'] for v in replay['banks'].values())
    c.write(output,dict(status='complete',scope='Post-audit sensitivity; does not retroactively pass the failed singleton tolerance',
        maximum_observed_state_coordinate_difference=maximum,conservative_20_times_L2_noise_floor=20*np.sqrt(1024)*maximum,
        banks=results,source_sha256=c.sha(__file__)))
    print('Replay sensitivity complete',flush=True)


if __name__=='__main__':main()
