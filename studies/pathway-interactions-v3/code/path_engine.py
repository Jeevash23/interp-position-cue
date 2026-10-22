"""Exact final-position query, attention-output and MLP-output interventions."""
import numpy as np
import core as c


class PathEngine(c.Engine):
    def __init__(self,config):
        self.model_examples=0
        super().__init__(config)
        self.layers=config['capture_layers']

    def forward(self,rows,*args,**kwargs):
        self.model_examples+=len(rows)
        return super().forward(rows,*args,**kwargs)

    def probe(self,row,deltas,specs=None,banks=None,clean=False):
        import transformers.models.gpt2.modeling_gpt2 as gpt2
        torch=self.torch;n=len(deltas);assert 0<n<=8
        dd=np.zeros((8,12,1024),np.float32);dd[:n]=deltas
        specs=list(specs or [{} for _ in range(n)])+[{} for _ in range(8-n)]
        original=gpt2.eager_attention_forward;observed={};mlps={};errors=np.zeros(8);handles=[]
        def target(spec,key,li):
            return banks[spec.get('bank','zero')][key][li]
        def wrapped(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
            out,prob=original(module,query,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
            layer=module.layer_idx
            if layer not in self.layers:return out,prob
            li=self.layers.index(layer);qq=query;changed=[]
            for b,s in enumerate(specs):
                hs=[h for l,h in s.get('q',[]) if l==layer]
                if hs:
                    if qq is query:qq=query.clone()
                    values=target(s,'q',li)[hs];qq[b,hs,-1]=values
                    errors[b]=max(errors[b],float((qq[b,hs,-1]-values).abs().max()))
                    changed.append((b,hs))
            if changed:
                computed,newprob=original(module,qq,key,value,attention_mask,scaling=scaling,dropout=dropout,**kwargs)
                out=out.clone();prob=prob.clone()
                for b,hs in changed:
                    out[b,-1,hs]=computed[b,-1,hs];prob[b,hs,-1]=newprob[b,hs,-1]
            for b,s in enumerate(specs):
                hs=[h for l,h in s.get('attn',[]) if l==layer]
                if hs:
                    out=out.clone();values=target(s,'z',li)[hs];out[b,-1,hs]=values
                    errors[b]=max(errors[b],float((out[b,-1,hs]-values).abs().max()))
            observed[layer]={'q':qq[:,:,-1].detach().cpu().numpy(),'z':out[:,-1].detach().cpu().numpy()}
            return out,prob
        for layer in self.layers:
            def mlp_hook(module,args,out,layer=layer):
                li=self.layers.index(layer)
                for b,s in enumerate(specs):
                    if layer in s.get('mlp',[]):
                        out=out.clone();values=target(s,'m',li);out[b,-1]=values
                        errors[b]=max(errors[b],float((out[b,-1]-values).abs().max()))
                mlps[layer]=out[:,-1].detach().cpu().numpy()
                return out
            handles.append(self.model.transformer.h[layer].mlp.register_forward_hook(mlp_hook))
        gpt2.eager_attention_forward=wrapped
        try:
            out=self.forward([row]*8,clean=clean,deltas=dd)
            return dict(logits=out['logits'][:n],state=out['states'][:n,-1],source=out['source'][:n],
                        q=np.stack([observed[l]['q'][:n] for l in self.layers],1),
                        z=np.stack([observed[l]['z'][:n] for l in self.layers],1),
                        m=np.stack([mlps[l][:n] for l in self.layers],1),write_error=errors[:n])
        finally:
            gpt2.eager_attention_forward=original
            for h in handles:h.remove()

    def banks(self,natural,labels):
        return {label:{k:self.torch.as_tensor(natural[k][i],device=self.device) for k in ['q','z','m']}
                for i,label in enumerate(labels)}


def source_bank(rows,sources,sc,sb):
    raw={r['id']:v for r,v in zip(sources,sc['source']-sb['source'])};normalized={}
    for r in rows:
        reference=np.linalg.norm(raw[r['norm_reference_id']],axis=1,keepdims=True)
        for sid in r['source_ids'].values():
            v=raw[sid];norm=np.linalg.norm(v,axis=1,keepdims=True);assert np.all(norm>0)
            expected=v*(reference/norm)
            if sid in normalized:assert np.array_equal(normalized[sid],expected)
            normalized[sid]=expected
    return raw,normalized


def sites(mask,selection,cfg):
    return dict(q=cfg['consumer_heads'] if mask&1 else [],attn=selection['attention_heads'] if mask&2 else [],mlp=selection['mlp_blocks'] if mask&4 else [])


def natural_label(align,dose):return f'{align}@{dose:g}'


def factorial_plan(selection,cfg):
    plan=[dict(id='zero',kind='zero',mask=0,dose=0.,align='none'),dict(id='zero_write',kind='zero',mask=0,dose=0.,align='none')]
    for dose in cfg['strengths']:
        for align in ['matched','opposite']:
            for kind,masks in [('clamp',range(8)),('restore',range(1,8))]:
                for mask in masks:
                    plan.append(dict(id=f'{kind}:{mask}:{natural_label(align,dose)}',kind=kind,mask=mask,dose=dose,align=align,
                        bank='zero' if kind=='clamp' else natural_label(align,dose),**sites(mask,selection,cfg)))
            for control in ['attention','mlp']:
                sham=dict(q=cfg['consumer_heads'],attn=selection['control_attention_heads'] if control=='attention' else [],
                          mlp=selection['control_mlp_blocks'] if control=='mlp' else [])
                for kind in ['clamp','restore']:
                    plan.append(dict(id=f'{kind}:sham_{control}:{natural_label(align,dose)}',kind=kind,mask=-1,dose=dose,align=align,
                                     bank='zero' if kind=='clamp' else natural_label(align,dose),**sham))
            for kind in ['clamp','restore']:
                plan.append(dict(id=f'{kind}:all:{natural_label(align,dose)}',kind=kind,mask=-2,dose=dose,align=align,
                    bank='zero' if kind=='clamp' else natural_label(align,dose),q=[],attn=[[l,h] for l in cfg['capture_layers'] for h in range(16)],mlp=cfg['capture_layers']))
    assert len(plan)==86
    return plan


def candidate_sites(cfg):
    known={tuple(x) for x in cfg['consumer_heads']}
    return [dict(id=f'A{l}.{h}',attn=[[l,h]],mlp=[]) for l in cfg['capture_layers'] for h in range(16) if (l,h) not in known]+[dict(id=f'M{l}',attn=[],mlp=[l]) for l in cfg['capture_layers']]


def screen_plan(cfg):
    plan=[dict(id='zero',kind='zero',mask=0,dose=0.,align='none'),dict(id='zero_write',kind='zero',mask=0,dose=0.,align='none')]
    for align in ['matched','opposite']:
        for label,q in [('full',[]),('Q',cfg['consumer_heads'])]:
            plan.append(dict(id=f'{label}:{align}',kind='clamp',dose=1.,align=align,bank='zero',q=q,attn=[],mlp=[]))
        for candidate in candidate_sites(cfg):
            plan.append(dict(id=f'{candidate["id"]}:{align}',candidate=candidate['id'],kind='clamp',dose=1.,align=align,bank='zero',
                             q=cfg['consumer_heads'],attn=candidate['attn'],mlp=candidate['mlp']))
    assert len(plan)==574
    return plan
