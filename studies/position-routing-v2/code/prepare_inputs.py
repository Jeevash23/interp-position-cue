"""Prepare a fixed positional-content factorial and independent donor/recipient name units."""
import re
import numpy as np
import core as c

EXTRA_NAMES='Aaron Abigail Adrian Alex Alexander Amanda Angela Anthony Arthur Austin Barry Benjamin Beth Betty Bill Billy Brad Brandon Brenda Bruce Carl Carlos Catherine Charlotte Chloe Christina Christine Christopher Cindy Craig Daniel Danny Dean Deborah Dennis Derek Diane Donald Donna Dorothy Douglas Dylan Earl Ed Elaine Elena Ellen Emily Ernest Eugene Evelyn Felix Francis Fred Gabriel Gary Gavin Geoffrey Gerald Gilbert Gloria Gordon Greg Gregory Harold Harry Heather Helen Howard Hugh Ian Isabel Ivan Jack James Janet Janice Jasmine Jean Jeff Jeffrey Jeremy Jerry Jesse Jessica Jim Jimmy Joan Joe Joel John Jonathan Jordan Joyce Juan Judith Judy Justin Kate Katherine Kathleen Kathy Katie Keith Kenneth Kurt Kyle Larry Laura Lawrence Lee Leo Leonard Leslie Lewis Logan Louis Louise Lucas Lucy Luke Marcus Margaret Maria Marie Marilyn Mark Martha Martin Mary Matthew Maureen Max Megan Melissa Mia Molly Monica Nancy Naomi Nathan Neil Nicholas Nick Norman Oliver Oscar Owen Pamela Pat Patricia Patrick Paul Paula Peggy Philip Phillip Rachel Ralph Randy Raymond Richard Rick Rita Rob Roger Ronald Ron Rosa Roy Ruby Russell Ruth Ryan Sally Sandra Scott Sebastian Shane Sharon Shirley Simon Sophie Stanley Stephanie Stephen Steven Stuart Sue Susan Suzanne Ted Teresa Terry Tim Timothy Tina Todd Tracy Trevor Tyler Valerie Vanessa Vernon Victoria Vincent Virginia Walter Wayne William Willie Zach'.split()
CONTEXTS=[['airport','camera'],['apartment','wallet'],['auditorium','umbrella'],['cabin','watch'],['church','newspaper'],['farm','bowl'],['hospital','notebook'],['restaurant','suitcase']]
TEMPLATES=dict(F10='In the {place}, {a} and {b} talked. {s} handed {article} {obj} to',F11='While {a} and {b} stood in the {place}, {s} passed {article} {obj} to')


def render(names,context,role,order,family):
    io,subject=(names[0],names[1]) if role==0 else (names[1],names[0]);a,b=(io,subject) if order==0 else (subject,io)
    place,obj=context;fields=dict(place=place,obj=obj,a=a,b=b,s=subject,article='an' if obj[0] in 'aeiou' else 'a')
    clean=TEMPLATES[family].format(**fields);fields['s']=names[2]
    return dict(clean_text=clean,corrupted_text=TEMPLATES[family].format(**fields),io=io,subject=subject,alternative=names[2],
        order=order,role=role,place=place,object=obj,family=family)


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(c.MODEL),local_files_only=True,trust_remote_code=False)
    previous=c.ROOT.parent/'fixed-signal-v1';pilot=c.read(previous/'data/pilot-qualified.json');confirmation=c.read(previous/'data/confirmation-qualified.json')
    oldnames={r[k] for b in [pilot,confirmation] for r in b['sources'] for k in ['io','subject','alternative']}
    pool=sorted(set(EXTRA_NAMES+confirmation['eligible_names'])-oldnames)
    eligible=[name for name in pool if len(tok.encode(' '+name,add_special_tokens=False))==1]
    rng=np.random.default_rng(2026100813);rng.shuffle(eligible);assert len(eligible)>=48,len(eligible)
    fresh=eligible[:48];units=[dict(recipient=fresh[i:i+3],foreign=fresh[i+3:i+6]) for i in range(0,48,6)]
    dev=[]
    for ni in range(4):
        def triple(split):
            r=next(r for r in pilot['sources'] if r['split']==split and r['name_index']==ni and r['role']==0)
            return [r['io'],r['subject'],r['alternative']]
        dev.append(dict(recipient=triple('development'),foreign=triple('holdout')))
    oldctx={(r['place'],r['object']) for b in [pilot,confirmation] for r in b['recipients']+b['sources']}
    assert not oldctx&{tuple(x) for x in CONTEXTS};assert not set(fresh)&oldnames
    devctx=[]
    for ci in range(4):
        r=next(r for r in pilot['recipients'] if r['split']=='development' and r['context_index']==ci);devctx.append([r['place'],r['object']])
    rows=[];sources=[]
    for split,uu,contexts in [('development',dev,devctx),('confirmation',units,CONTEXTS)]:
        for ni,u in enumerate(uu):
            for role in [0,1]:
                group=f'{split}:unit{ni}:role{role}'
                for kind,names in [('same',u['recipient']),('foreign',u['foreign'])]:
                    for order in [0,1]:sources.append(dict(id=f'{group}:{kind}:order{order}',split=split,name_index=ni,context_index=None,
                        source_kind=kind,**render(names,['atrium','parcel'],role,order,'F10')))
                for ci,context in enumerate(contexts):
                    for family in TEMPLATES:
                        for order in [0,1]:rows.append(dict(id=f'{group}:context{ci}:{family}:order{order}',split=split,name_index=ni,context_index=ci,
                            source_ids={f'{kind}_{align}':f'{group}:{kind}:order{order if align=="matched" else 1-order}' for kind in ['same','foreign'] for align in ['matched','opposite']},
                            norm_reference_id=f'{group}:same:order0',**render(u['recipient'],context,role,order,family)))
    for row in rows+sources:
        a=tok(row['clean_text'],add_special_tokens=False,return_offsets_mapping=True);b=tok(row['corrupted_text'],add_special_tokens=False)
        clean=[50256]+a['input_ids'];bad=[50256]+b['input_ids'];changed=[i for i,(x,y) in enumerate(zip(clean,bad)) if x!=y]
        assert len(clean)==len(bad) and len(changed)==1;s2=changed[0]
        ids={k:tok.encode(' '+row[name],add_special_tokens=False)[0] for name,k in [('io','io_id'),('subject','subject_id'),('alternative','alt_id')]}
        assert clean[s2]==ids['subject_id'] and bad[s2]==ids['alt_id']
        row.update(**ids,tokens_clean=clean,tokens_corrupt=bad,s2=s2,answer_position=len(clean)-1,
            io_position=clean.index(ids['io_id']),s1_position=clean.index(ids['subject_id']),token_offsets=a['offset_mapping'])
    oldprompts={r[field] for b in [pilot,confirmation] for key in ['sources','recipients'] for r in b[key] for field in ['clean_text','corrupted_text']}
    assert not oldprompts&{r[field] for r in rows+sources if r['split']=='confirmation' for field in ['clean_text','corrupted_text']}
    # The random control heads are fixed before any follow-up model output.
    cfg=c.read(c.ROOT/'protocol/config.json');controls=[[],[]]
    for layer,head in cfg['consumer_heads']:
        picks=rng.choice([h for h in range(16) if h!=head],size=2,replace=False)
        for k,pick in enumerate(picks):controls[k].append([layer,int(pick)])
    rotations=[]
    for layer,head in cfg['consumer_heads']:
        q,r=np.linalg.qr(rng.normal(size=(64,64)));q*=np.sign(np.diag(r))[None,:];rotations.append(q.astype(np.float32))
    np.savez_compressed(c.ROOT/'data/query-rotations.npz',rotations=np.array(rotations))
    c.write(c.ROOT/'data/inputs.json',dict(created_utc=c.now(),recipients=rows,sources=sources,development_units=dev,
        confirmation_units=units,confirmation_contexts=CONTEXTS,eligible_names=eligible,selection_seed=2026100813,
        control_head_sets=controls,AI08_confirmation_names_and_context_pairs_disjoint=True,confirmation_exact_AI08_prompt_overlap=0,
        old_study_manifest_sha256=c.sha(previous/'EVIDENCE-MANIFEST.json')))
    print('Prepared',len(rows),'recipients and',len(sources),'sources;',len(eligible),'eligible fresh names; controls',controls,flush=True)


if __name__=='__main__':main()
