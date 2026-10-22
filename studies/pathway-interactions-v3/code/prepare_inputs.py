"""Outcome-independent inputs for a bounded pathway screen and factorial confirmation."""
import numpy as np
import core as c
from input_helpers import TEMPLATES as OLD_TEMPLATES

NEW_TEMPLATES = {
    'F12': 'After {a} and {b} arrived at the {place}, {s} gave {article} {obj} to',
    'F13': 'Earlier, {a} met {b} at the {place}. Then {s} gave {article} {obj} to',
}
CONTEXTS = [['stadium','helmet'],['theater','guitar'],['beach','towel'],['kitchen','spoon'],
            ['bedroom','hat'],['gallery','brush'],['warehouse','plate'],['classroom','pen']]


def make(names, context, role, order, family, corruption):
    io, sub = (names[0],names[1]) if role == 0 else (names[1],names[0])
    a,b = (io,sub) if order == 0 else (sub,io)
    place,obj = context
    template = (OLD_TEMPLATES | NEW_TEMPLATES)[family]
    fields = dict(a=a,b=b,s=sub,place=place,obj=obj,article='an' if obj[0] in 'aeiou' else 'a')
    clean = template.format(**fields)
    fields['s'] = names[2] if corruption == 'third_name' else io
    return dict(clean_text=clean,corrupted_text=template.format(**fields),io=io,subject=sub,
                alternative=names[2],place=place,object=obj,role=role,order=order,
                family=family,corruption=corruption)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(c.MODEL),local_files_only=True,trust_remote_code=False)
    oldroot = c.ROOT.parent/'position-routing-v2';old = c.read(oldroot/'data/inputs.json')
    used = {r[k] for r in old['recipients']+old['sources'] for k in ['io','subject','alternative']}
    pool = sorted(set(old['eligible_names'])-used)
    rng = np.random.default_rng(2026100822);rng.shuffle(pool);assert len(pool)>=24
    fresh = [pool[i:i+3] for i in range(0,24,3)]
    dev = [u['recipient'] for u in old['development_units']]
    devctx = [[r['place'],r['object']] for r in old['recipients'] if r['split']=='development' and r['name_index']==0 and r['role']==0 and r['family']=='F10' and r['order']==0]
    rows=[];sources=[]
    for split,units,contexts,families in [('development',dev,devctx,list(OLD_TEMPLATES)),('confirmation',fresh,CONTEXTS,list(NEW_TEMPLATES))]:
        for ni,names in enumerate(units):
            for role in [0,1]:
                for corruption in ['third_name','repeat_io']:
                    group=f'{split}:n{ni}:r{role}:{corruption}'
                    for order in [0,1]:
                        sources.append(dict(id=f'{group}:source{order}',split=split,name_index=ni,context_index=None,
                                            **make(names,['atrium','parcel'],role,order,'F10',corruption)))
                    for ci,context in enumerate(contexts):
                        for family in families:
                            for order in [0,1]:
                                rows.append(dict(id=f'{group}:c{ci}:{family}:o{order}',split=split,name_index=ni,context_index=ci,
                                    source_ids={'matched':f'{group}:source{order}','opposite':f'{group}:source{1-order}'},
                                    norm_reference_id=f'{group}:source0',**make(names,context,role,order,family,corruption)))
    for r in rows+sources:
        ids={k:tok.encode(' '+r[n],add_special_tokens=False) for k,n in [('io_id','io'),('subject_id','subject'),('alt_id','alternative')]}
        assert all(len(v)==1 for v in ids.values());ids={k:v[0] for k,v in ids.items()}
        clean=[50256]+tok.encode(r['clean_text'],add_special_tokens=False);bad=[50256]+tok.encode(r['corrupted_text'],add_special_tokens=False)
        changed=[i for i,(a,b) in enumerate(zip(clean,bad)) if a!=b]
        assert len(clean)==len(bad) and len(changed)==1
        s2=changed[0];assert clean[s2]==ids['subject_id']
        assert bad[s2]==ids['alt_id' if r['corruption']=='third_name' else 'io_id']
        r.update(**ids,tokens_clean=clean,tokens_corrupt=bad,s2=s2,io_position=clean.index(ids['io_id']),s1_position=clean.index(ids['subject_id']),answer_position=len(clean)-1)
    old_banks=[old,c.read(c.ROOT.parent/'fixed-signal-v1/data/pilot-qualified.json'),c.read(c.ROOT.parent/'fixed-signal-v1/data/confirmation-qualified.json')]
    oldrows=[r for b in old_banks for key in ['recipients','sources'] for r in b[key]]
    oldnames={r[k] for r in oldrows for k in ['io','subject','alternative']}
    assert not {n for u in fresh for n in u}&oldnames
    assert not {tuple(x) for x in CONTEXTS}&{(r['place'],r['object']) for r in oldrows}
    oldstrings={r[k] for r in oldrows for k in ['clean_text','corrupted_text']}
    assert not {r[k] for r in rows+sources if r['split']=='confirmation' for k in ['clean_text','corrupted_text']}&oldstrings
    screen=[r['id'] for r in rows if r['split']=='development' and r['family']=='F10' and r['context_index'] in [0,3] and r['corruption']=='third_name']
    assert len(screen)==32 and sum(r['split']=='development' for r in rows)==256 and sum(r['split']=='confirmation' for r in rows)==1024
    c.write(c.ROOT/'data/inputs.json',dict(created_utc=c.now(),recipients=rows,sources=sources,development_names=dev,confirmation_names=fresh,
        confirmation_contexts=CONTEXTS,screen_ids=screen,selection_seed=2026100822,new_confirmation_names=24,
        fresh_confirmation_templates=NEW_TEMPLATES,prior_manifests={p.name:c.sha(p/'EVIDENCE-MANIFEST.json') for p in [oldroot,c.ROOT.parent/'fixed-signal-v1']}))
    print('Prepared',len(rows),'recipient cases,',len(sources),'sources;',len(screen),'screen cases')


if __name__=='__main__':main()
