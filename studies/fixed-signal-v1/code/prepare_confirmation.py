"""Fixed larger bank: new AI-08 groups, with historical prompt-overlap checks."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import core as c

NAMES='Adam Amy Andrew Anna Anne Ben Brian Charles Chris Claire Dan Daniel Diana Edward Elizabeth Emily Eric Ethan Eva George Hannah Harry Helen Ian Jack Jacob James Jane Jason Jennifer Jessica Joseph Josh Julie Karen Kelly Ken Kim Linda Lisa Mark Martin Matt Michael Michelle Mike Nancy Natalie Nathan Nick Nicole Paul Patrick Rachel Rebecca Richard Robert Ryan Sam Sarah Scott Sean Simon Sophie Steve Steven Susan Tom Tony William'.split()
CONTEXTS=[['bakery','ribbon'],['balcony','coin'],['cafeteria','envelope'],['cinema','key'],['courtyard','map'],['depot','bottle'],['factory','photograph'],['garage','package'],['harbor','bracelet'],['laboratory','painting'],['lobby','scarf'],['workshop','box']]


def main():
    from transformers import AutoTokenizer
    sys.path.insert(0,str(c.PROJECT/'02-Code'));from prepare import build
    t=AutoTokenizer.from_pretrained(c.MODEL,local_files_only=True);old=c.read(c.ROOT/'data/pilot-qualified.json')
    used={r[k] for r in old['sources'] for k in ['io','subject','alternative']}
    eligible=[n for n in NAMES if n not in used and len(t.encode(' '+n,add_special_tokens=False))==1]
    rng=np.random.default_rng(2026100810);rng.shuffle(eligible);names=eligible[:36];assert len(names)==36
    cfg=c.read(c.PROJECT/'03-Data/config/pilot.json');cfg['splits']={'confirmation':dict(name_triples=[names[i:i+3] for i in range(0,36,3)],contexts=CONTEXTS)}
    data=build(cfg);sources=data['sources.jsonl'];recipients=data['recipients.jsonl']
    for row in sources+recipients:
        a=t(row['clean_text'],add_special_tokens=False,return_offsets_mapping=True);b=t(row['corrupted_text'],add_special_tokens=False)
        clean=[t.bos_token_id]+a['input_ids'];bad=[t.bos_token_id]+b['input_ids'];span=list(re.finditer(r'\b'+row['subject']+r'\b',row['clean_text']))[1].span()
        hit=[i+1 for i,(left,right) in enumerate(a['offset_mapping']) if right>span[0] and left<span[1]];assert len(hit)==1;s2=hit[0]
        assert len(clean)==len(bad) and [i for i,(x,y) in enumerate(zip(clean,bad)) if x!=y]==[s2]
        row.update(tokens_clean=clean,tokens_corrupt=bad,s2=s2,answer_position=len(clean)-1,token_offsets=a['offset_mapping'],tokenization_status='qualified',
            name_index=int(row['group_id'].split(':names')[1].split(':')[0]),context_index=None if 'block_id' not in row else int(row['block_id'].rsplit('context',1)[1]))
        for key,field in [('io','io_id'),('subject','subject_id'),('alternative','alt_id')]:
            ids=t.encode(' '+row[key],add_special_tokens=False);assert len(ids)==1;row[field]=ids[0]
    assert not set(names)&used
    assert not {tuple(v) for v in CONTEXTS}&{(r['place'],r['object']) for r in old['recipients']+old['sources']}
    historical_texts=set();historical_names=set();hashes={}
    def visit(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():visit(v,k)
        elif isinstance(value,list):
            for v in value:visit(v,key)
        elif isinstance(value,str):
            if value.startswith(('In the ','While ')) and value.endswith(' to'):historical_texts.add(value)
            if key in ['io','subject','alternative','IO','S','S1','S2','a','b','s']:historical_names.add(value)
    # Release: the earlier project whose prompts and names this step excluded is not included; set
    # EARLIER_PROJECT_DIR to its folder to repeat the exclusion.
    ref=c.Path(c.os.environ['EARLIER_PROJECT_DIR'])
    dirs=[ref/'data']+[p/'data' for p in (ref/'studies').iterdir() if p.is_dir() and (p/'data').is_dir()]
    for folder in dirs:
        for p in sorted(folder.rglob('*')):
            if p.suffix not in ['.json','.jsonl'] or not p.is_file():continue
            hashes[str(p.relative_to(c.FIELD))]=c.sha(p)
            if p.suffix=='.json':visit(c.read(p))
            else:
                for line in p.read_text().splitlines():
                    if line.strip():visit(json.loads(line))
    duplicates=[r['id'] for r in sources+recipients if r['clean_text'] in historical_texts or r['corrupted_text'] in historical_texts]
    assert not duplicates,duplicates
    output=c.ROOT/'data/confirmation-qualified.json';assert not output.exists()
    c.write(output,dict(created_utc=c.now(),sources=sources,recipients=recipients,plans=data['interventions.jsonl'],selection_seed=2026100810,
        eligible_names=eligible,selected_names=names,contexts=CONTEXTS,source_context=cfg['source_context'],single_token_names=True,
        historical_exact_prompt_overlap=duplicates,historical_name_overlap=sorted(set(names)&historical_names),
        historical_texts_checked=len(historical_texts),historical_input_hashes=hashes,
        scope='New AI-08 groups; historical names can overlap and no pretraining-disjointness is claimed.'))
    print(json.dumps(dict(recipients=len(recipients),sources=len(sources),conditions=len(data['interventions.jsonl']),names=names,
        historical_name_overlap=sorted(set(names)&historical_names),historical_exact_prompt_overlap=duplicates)))


if __name__=='__main__':main()
