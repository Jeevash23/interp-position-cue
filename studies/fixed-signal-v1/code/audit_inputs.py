"""Retokenize all study inputs and verify bank separation and historical-file fingerprints."""
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
import core as c


def strings(value):
    if isinstance(value,str):yield value
    elif isinstance(value,dict):
        for v in value.values():yield from strings(v)
    elif isinstance(value,list):
        for v in value:yield from strings(v)


def main():
    output=c.ROOT/'analysis/INPUT-AUDIT.json';assert not output.exists()
    tokenizer=AutoTokenizer.from_pretrained(str(c.MODEL),local_files_only=True,trust_remote_code=False,use_fast=True)
    a=c.read(c.ROOT/'data/pilot-qualified.json');b=c.read(c.ROOT/'data/confirmation-qualified.json');banks={};total=0
    for bundle in [a,b]:
        for row in bundle['sources']+bundle['recipients']:
            total+=1;group=banks.setdefault(row['split'],dict(names=set(),contexts=set(),prompts=set(),sources=0,recipients=0))
            group['names'].update([row['subject'],row['io'],row['alternative']]);group['prompts'].update([row['clean_text'],row['corrupted_text']])
            group['sources' if row['context_index'] is None else 'recipients']+=1
            if row['context_index'] is not None:group['contexts'].add((row['place'],row['object']))
            for textfield,tokenfield in [('clean_text','tokens_clean'),('corrupted_text','tokens_corrupt')]:
                got=tokenizer(row[textfield],add_special_tokens=False,return_offsets_mapping=True)
                assert [50256]+got['input_ids']==row[tokenfield]
                if textfield=='clean_text':assert [list(v) for v in got['offset_mapping']]==row['token_offsets']
            for name,token in [('subject','subject_id'),('io','io_id'),('alternative','alt_id')]:
                assert tokenizer.encode(' '+row[name],add_special_tokens=False)==[row[token]]
            start,end=row['token_offsets'][row['s2']-1]
            assert row['clean_text'][start:end].strip()==row['subject']
    keys=list(banks)
    for i,x in enumerate(keys):
        for y in keys[i+1:]:
            for field in ['names','contexts','prompts']:assert not banks[x][field]&banks[y][field],(x,y,field)
    history=set()
    for rel,digest in b['historical_input_hashes'].items():
        p=c.FIELD/rel;assert c.sha(p)==digest
        data=[json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.suffix=='.jsonl' else c.read(p)
        history.update(strings(data))
    assert not banks['confirmation']['prompts']&history
    overlap=[name for name in sorted(banks['confirmation']['names']) if any(re.search(r'\b'+re.escape(name)+r'\b',s) for s in history)]
    c.write(output,dict(status='passed',created_utc=c.now(),retokenized_pairs=total,
        banks={k:{f:sorted(v) if isinstance(v,set) else v for f,v in g.items() if f!='prompts'} for k,g in banks.items()},
        pairwise_AI08_names_contexts_and_exact_prompts_disjoint=True,historical_files_unchanged=len(b['historical_input_hashes']),
        confirmation_exact_historical_prompt_overlap=0,confirmation_names_in_historical_strings=overlap,
        source_sha256=c.sha(__file__)))
    print('Input audit passed:',total,'pairs;',len(overlap),'confirmation names occur in historical strings',flush=True)


if __name__=='__main__':main()
