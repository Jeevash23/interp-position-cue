"""Additional engineering comparison with V2 on unchanged development cases.

Added while confirmation was running; this is not a new confirmatory hypothesis
or selection criterion. It only checks corresponding previously measured states.
"""
import numpy as np
import core as c


def main():
    old=c.ROOT.parent/'position-routing-v2/runs/development';new=c.ROOT/'runs/development'
    oldrows=c.read(old/'recipients.json');newrows=c.read(new/'recipients.json');plans=c.read(new/'conditions.json');ix={p['id']:i for i,p in enumerate(plans)}
    meta=c.read(old/'conditions.json');oldlabels=[p['id'] for p in meta if p['row_id']==oldrows[0]['id']];oi={label:i for i,label in enumerate(oldlabels)}
    lookup={(r['clean_text'],r['corrupted_text'],r['io']):i for i,r in enumerate(oldrows)}
    mappings={'zero':'zero'}
    for align in ['matched','opposite']:
        mappings[f'clamp:0:{align}@1']=f'same_{align}'
        mappings[f'clamp:1:{align}@1']=f'same_{align}:clamp_q'
        mappings[f'restore:1:{align}@1']=f'same_{align}:query_only'
    records=[]
    for i,row in enumerate(newrows):
        if row['corruption']!='third_name':continue
        j=lookup[(row['clean_text'],row['corrupted_text'],row['io'])]
        with np.load(new/f'row-{i:04d}.npz') as a,np.load(old/f'recipient-{j:04d}.npz') as b:
            bt=b['logits'][:,[row['io_id'],row['subject_id']]];bs=b['states'][:,-1]
            for newlabel,oldlabel in mappings.items():
                at=a['target_logits'][ix[newlabel]];astate=a['state'][ix[newlabel]]
                records.append(dict(row_id=row['id'],condition=newlabel,target_error=float(np.max(np.abs(at-bt[oi[oldlabel]]))),
                                    state_error=float(np.max(np.abs(astate-bs[oi[oldlabel]])))))
    maximum=max(max(r['target_error'],r['state_error']) for r in records)
    output=c.ROOT/'analysis/PREVIOUS-ADAPTER-BRIDGE.json';assert not output.exists()
    c.write(output,dict(status='passed' if maximum<=.0005 else 'failed',created_utc=c.now(),source_sha256=c.sha(__file__),
        scope='Engineering diagnostic on 128 reused development cases and seven corresponding conditions; not a fresh confirmation claim.',
        added_after_confirmation_started=True,used_confirmation_outcomes=False,maximum_error=maximum,bitwise_equal=maximum==0.,comparisons=records))
    assert maximum<=.0005
    print('Previous-adapter bridge:',len(records),'comparisons; maximum error',maximum,flush=True)


if __name__=='__main__':main()
