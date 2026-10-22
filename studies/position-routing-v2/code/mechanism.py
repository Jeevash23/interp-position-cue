"""Path interventions on the final-position Q, K, V or attention probabilities."""
import numpy as np
import core as c


class MechanismEngine(c.Engine):
    def __init__(self,config):
        self.model_examples=0
        super().__init__(config)
        self.capture_layers=config['capture_layers'];self.consumer_heads=config['consumer_heads']

    def forward(self,rows,*args,**kwargs):
        self.model_examples+=len(rows)
        return super().forward(rows,*args,**kwargs)

    def probe(self,row,deltas,specs=None,baseline=None,query_banks=None,clean=False,capture_baseline=False):
        """Always use eight copies of one prompt; pad unused conditions and record their cost."""
        import transformers.models.gpt2.modeling_gpt2 as gpt2
        torch=self.torch;n=len(deltas);assert 0<n<=8
        specs=list(specs or [dict(mode='none') for _ in range(n)])
        dd=np.zeros((8,12,1024),np.float32);dd[:n]=deltas;specs+= [dict(mode='none') for _ in range(8-n)]
        original=gpt2.eager_attention_forward;observed={};captured={}
        def wrapped(module,query,key,value,attention_mask,scaling=None,dropout=0.0,**kwargs):
            output,prob=original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
            layer=module.layer_idx
            if layer not in self.capture_layers:return output,prob
            li=self.capture_layers.index(layer);qq=query;kk=key;vv=value;pp=prob
            interventions=[];qchanged=False;kchanged=False
            for b,spec in enumerate(specs):
                mode=spec['mode'];heads=[h for l,h in spec.get('heads',[]) if l==layer]
                if not heads or mode=='none':continue
                assert baseline is not None
                interventions.append((b,heads,mode))
                if mode in ['q','query_only','rotated_query']:
                    if qq is query:qq=query.clone()
                    for h in heads:
                        if mode=='q':target=baseline[layer]['q'][h]
                        elif mode=='query_only':target=query_banks[spec['bank']][li,h]
                        else:
                            hi=self.consumer_heads.index([layer,h]);delta=query_banks[spec['bank']][li,h]-baseline[layer]['q'][h]
                            target=baseline[layer]['q'][h]+delta@query_banks['rotations'][hi]
                        qq[b,h,-1]=target
                    qchanged=True
                elif mode=='k':
                    if kk is key:kk=key.clone()
                    for h in heads:kk[b,h]=baseline[layer]['k'][h]
                    kchanged=True
                elif mode in ['v','pv']:
                    if vv is value:vv=value.clone()
                    for h in heads:vv[b,h]=baseline[layer]['v'][h]
            if qchanged or kchanged:
                _,computed=original(module,qq,kk,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
                pp=prob.clone()
                for b,heads,mode in interventions:
                    if mode in ['q','k','query_only','rotated_query']:pp[b,heads,-1]=computed[b,heads,-1]
            for b,heads,mode in interventions:
                if mode in ['p','pv']:
                    if pp is prob:pp=prob.clone()
                    pp[b,heads,-1]=baseline[layer]['p'][heads]
            if interventions:
                recalculated=torch.matmul(pp,vv).transpose(1,2);edited=output.clone()
                for b,heads,mode in interventions:
                    if mode=='z':edited[b,-1,heads]=baseline[layer]['z'][heads]
                    else:edited[b,-1,heads]=recalculated[b,-1,heads]
                output=edited
            observed[layer]=dict(q=qq[:,:,-1].detach().cpu().numpy(),p=pp[:,:,-1].detach().cpu().numpy(),z=output[:,-1].detach().cpu().numpy())
            if capture_baseline:
                captured[layer]=dict(q=query[0,:,-1].detach().clone(),k=key[0].detach().clone(),v=value[0].detach().clone(),
                    p=prob[0,:,-1].detach().clone(),z=output[0,-1].detach().clone())
            return output,pp
        gpt2.eager_attention_forward=wrapped
        try:
            out=self.forward([row]*8,clean=clean,deltas=dd)
            for key in ['q','p','z']:out[key]=np.stack([observed[layer][key] for layer in self.capture_layers],axis=1)
            return {k:v[:n] for k,v in out.items()},captured
        finally:gpt2.eager_attention_forward=original


def signal_bank(rows,sources,clean,corrupt):
    cache={r['id']:d for r,d in zip(sources,clean['source']-corrupt['source'])}
    normalized={}
    for row in rows:
        reference=np.linalg.norm(cache[row['norm_reference_id']],axis=1,keepdims=True)
        for sid in row['source_ids'].values():
            if sid not in normalized:
                d=cache[sid];norm=np.linalg.norm(d,axis=1,keepdims=True);assert np.all(norm>0)
                normalized[sid]=d*(reference/norm)
    return cache,normalized


def condition_plan(row,heads,controls):
    """45 saved conditions; five natural states are measured in the initial shared batch."""
    sourcekeys=['same_matched','same_opposite','foreign_matched','foreign_opposite']
    plans=[dict(id='zero',source=None,scale='zero',mode='none')]
    plans += [dict(id=key,source=key,scale='normalized',mode='none') for key in sourcekeys]
    plans += [dict(id='raw_'+key,source=key,scale='raw',mode='none') for key in sourcekeys]
    plans += [dict(id='recipient_self',source=None,scale='self',mode='none'),dict(id='zero_write',source=None,scale='zero',mode='none')]
    for key in sourcekeys:
        for mode in ['q','k','v','p','z']:
            plans.append(dict(id=f'{key}:clamp_{mode}',source=key,scale='normalized',mode=mode,heads=heads))
        for i,hh in enumerate(controls):
            plans.append(dict(id=f'{key}:control{i}_q',source=key,scale='normalized',mode='q',heads=hh))
        plans.append(dict(id=f'{key}:query_only',source=None,scale='zero',mode='query_only',bank=key,heads=heads))
    for key in ['same_matched','same_opposite']:
        plans.append(dict(id=f'{key}:rotated_query',source=None,scale='zero',mode='rotated_query',bank=key,heads=heads))
    assert len(plans)==45;return plans
