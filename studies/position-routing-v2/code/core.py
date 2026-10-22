"""Isolated offline GPT-2 adapter for AI-08. No earlier pipeline is imported."""
from __future__ import annotations
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
FIELD=PROJECT.parent
MODEL=Path(os.environ.get('MODELS_DIR',PROJECT/'models'))/'gpt2-medium'   # release: see models/README.md
LOCK=Path(os.environ.get('GPU_LOCK',PROJECT/'gpu.lock'))   # release: one lock file for GPU jobs on a machine
for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE']:os.environ[k]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'


def read(p):return json.loads(Path(p).read_text())
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
    return h.hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(p)
def array_hash(a):
    import numpy as np
    a=np.ascontiguousarray(a);return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def guard():
    m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());avail=int(m['MemAvailable'].split()[0])*1024
    disk=shutil.disk_usage(ROOT).free
    if avail<3*2**30 or disk<4*2**30:raise RuntimeError('Resource guard: need 3 GiB available memory and 4 GiB disk')
    return dict(available_memory_bytes=avail,free_disk_bytes=disk)
def snapshot(output,files):
    hashes={}
    for p in files:
        p=Path(p).resolve();rel=p.relative_to(ROOT);dest=Path(output)/'snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);hashes[str(rel)]=sha(p)
    return hashes


class Engine:
    def __init__(self,config):
        guard();self.config=config;self.lock=None
        if config['device']=='cuda':
            self.lock=LOCK.open('r')
            try:fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Shared GPU lock is busy; no allocation attempted')
        import numpy as np
        import torch
        from transformers import AutoModelForCausalLM
        self.np=np;self.torch=torch;torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if config['device']=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('CUDA not accessible in this execution environment')
            free,total=torch.cuda.mem_get_info()
            if free<4*2**30:raise RuntimeError('GPU resource guard: need 4 GiB free')
        self.device=torch.device(config['device']);start=time.perf_counter()
        self.model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,trust_remote_code=False,
            dtype=torch.float32,attn_implementation='eager').to(self.device).eval()
        for p in self.model.parameters():p.requires_grad_(False)
        self.source=[tuple(h) for h in config['source_heads']];self.blocks=config['response_blocks']
        self.load_seconds=time.perf_counter()-start
        self.environment=dict(packages={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','scipy']},
            device=str(self.device),dtype='float32',threads=2,model_load_seconds=self.load_seconds,
            gpu=torch.cuda.get_device_name(0) if self.device.type=='cuda' else None)

    def forward(self,rows,clean=False,deltas=None,individual_delta=None):
        torch,np=self.torch,self.np
        assert rows and len({len(r['tokens_clean']) for r in rows})==1
        tokens=torch.tensor([r['tokens_clean' if clean else 'tokens_corrupt'] for r in rows],device=self.device)
        positions=torch.tensor([r['s2'] for r in rows],device=self.device);ix=torch.arange(len(rows),device=self.device)
        captured={};states={};handles=[]
        if deltas is not None:
            assert tuple(deltas.shape)==(len(rows),len(self.source),1024)
            dd=torch.as_tensor(deltas,dtype=torch.float32,device=self.device)
        for layer in sorted({h[0] for h in self.source}):
            projection=self.model.transformer.h[layer].attn.c_proj
            local=[(i,h) for i,(l,h) in enumerate(self.source) if l==layer]
            def before(module,args,local=local,layer=layer):
                z=args[0][ix,positions].reshape(len(rows),16,64)
                for i,h in local:captured[i]=(z[:,h]@module.weight[h*64:(h+1)*64]).detach()
                if individual_delta is not None and layer==individual_delta[0]:
                    # Independent intervention route: alter one head's pre-projection z.
                    value=args[0].clone();head=individual_delta[1]
                    value[ix,positions,head*64:(head+1)*64]+=torch.as_tensor(individual_delta[2],device=self.device)
                    return (value,)
            handles.append(projection.register_forward_pre_hook(before))
            if deltas is not None:
                d=dd[:,[i for i,h in local]].sum(1)
                def after(module,args,value,d=d):
                    out=value.clone();out[ix,positions]+=d;return out
                handles.append(projection.register_forward_hook(after))
        for layer in self.blocks:
            def observe(module,args,value,layer=layer):
                hidden=value[0] if isinstance(value,tuple) else value
                states[layer]=hidden[:,-1].detach()
            handles.append(self.model.transformer.h[layer].register_forward_hook(observe))
        try:
            with torch.inference_mode():
                logits=self.model(tokens,use_cache=False).logits[:,-1]
                return dict(logits=logits.cpu().numpy(),states=torch.stack([states[l] for l in self.blocks],1).cpu().numpy(),
                    source=torch.stack([captured[i] for i in range(len(self.source))],1).cpu().numpy())
        finally:
            for h in handles:h.remove()

    def measure(self,rows,clean=False,deltas=None,batch_size=None):
        np=self.np;batch_size=batch_size or self.config['batch_size'];values={}
        by_length={}
        for i,r in enumerate(rows):by_length.setdefault(len(r['tokens_clean']),[]).append(i)
        for length,ids in by_length.items():
            for start in range(0,len(ids),batch_size):
                select=ids[start:start+batch_size]
                out=self.forward([rows[i] for i in select],clean,None if deltas is None else deltas[select])
                for key,a in out.items():
                    if key not in values:values[key]=np.empty((len(rows),*a.shape[1:]),dtype=a.dtype)
                    values[key][select]=a
        return values

    def close(self):
        if self.lock is not None:fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN);self.lock.close();self.lock=None


def behavior(logits,rows):
    import numpy as np
    from scipy.special import logsumexp
    z=logits.astype(np.float64);ix=np.arange(len(rows));io=np.array([r['io_id'] for r in rows]);sub=np.array([r['subject_id'] for r in rows])
    logp=z-logsumexp(z,axis=1)[:,None];p=np.exp(logp)
    return dict(top1=z.argmax(1),correct=z.argmax(1)==io,margin=z[ix,io]-z[ix,sub],io_probability=p[ix,io],
                entropy=-np.sum(p*logp,axis=1))
