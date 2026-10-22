"""Select exactly two attention heads and one MLP using development screen only."""
import numpy as np
import core as c
from path_engine import candidate_sites


def main():
    run=c.ROOT/'runs/screen';assert c.read(run/'status.json')['status']=='complete'
    cfg=c.read(c.ROOT/'protocol/config.json');rows=c.read(run/'recipients.json');plan=c.read(run/'conditions.json');ix={p['id']:i for i,p in enumerate(plan)}
    margins=[]
    for i in range(len(rows)):
        with np.load(run/f'row-{i:04d}.npz') as z:margins.append(z['target_logits'].astype(float)[:,0]-z['target_logits'].astype(float)[:,1])
    margins=np.array(margins)
    def diff(key):return margins[:,ix[f'{key}:matched']]-margins[:,ix[f'{key}:opposite']]
    q=diff('Q');full=diff('full');rank=[]
    for candidate in candidate_sites(cfg):
        gain=q-diff(candidate['id']);groups=[float(gain[[r['name_index']==n for r in rows]].mean()) for n in range(4)]
        rank.append(dict(candidate,incremental_removal=float(gain.mean()),fraction_of_full=float(gain.mean()/full.mean()),name_means=groups,positive_name_units=sum(v>0 for v in groups)))
    rank.sort(key=lambda r:(-r['incremental_removal'],r['id']))
    heads=[r for r in rank if r['attn']][:2];mlps=[r for r in rank if r['mlp']][:1]
    selected=[h for r in heads for h in r['attn']];selected_m=[l for r in mlps for l in r['mlp']]
    rng=np.random.default_rng(2026100824);sham=[];known={tuple(x) for x in cfg['consumer_heads']+selected}
    for layer,head in selected:
        allowed=[h for h in range(16) if (layer,h) not in known and [layer,h] not in sham]
        sham.append([layer,int(rng.choice(allowed))])
    sham_m=[int(rng.choice([l for l in cfg['capture_layers'] if l not in selected_m]))]
    result=dict(created_utc=c.now(),source_sha256=c.sha(__file__),screen_only=True,screen_recipients=len(rows),candidate_count=len(rank),
        attention_heads=selected,mlp_blocks=selected_m,control_attention_heads=sham,control_mlp_blocks=sham_m,
        full_order_contrast=float(full.mean()),known_query_clamped_contrast=float(q.mean()),
        selected_screen_results=heads+mlps,selection_rule='Top two noncanonical attention heads and top one MLP by mean extra order-contrast removal while known queries are fixed. Ties by candidate id; no confirmation outcomes.',
        development_screen_support=all(r['incremental_removal']>0 and r['positive_name_units']>=3 for r in heads+mlps))
    output=c.ROOT/'analysis/SELECTION.json';assert not output.exists();c.write(output,result);c.write(c.ROOT/'analysis/SCREEN-RANKING.json',rank)
    print(result,flush=True)


if __name__=='__main__':main()
