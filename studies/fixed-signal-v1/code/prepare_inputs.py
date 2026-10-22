"""Qualify symbolic positions without measuring model outcomes."""
import json
import re
import sys
from pathlib import Path
import core as c


def main():
    from transformers import AutoTokenizer
    sys.path.insert(0,str(c.PROJECT/'02-Code'));from prepare import build
    config=c.read(c.PROJECT/'03-Data/config/pilot.json')
    tokenizer=AutoTokenizer.from_pretrained(c.MODEL,local_files_only=True)
    failures=[]
    for split,pools in config['splits'].items():
        for names in pools['name_triples']:
            for name in names:
                ids=tokenizer.encode(' '+name,add_special_tokens=False)
                if len(ids)!=1:failures.append(dict(split=split,name=name,tokens=ids))
    assert failures==[{'split':'holdout','name':'Uma','tokens':[471,2611]}]
    config['splits']['holdout']['name_triples'][2][2]='Nina'
    data=build(config);sources=data['sources.jsonl'];recipients=data['recipients.jsonl']
    for row in sources+recipients:
        clean=row['clean_text'];corrupt=row['corrupted_text']
        en=tokenizer(clean,add_special_tokens=False,return_offsets_mapping=True)
        ec=tokenizer(corrupt,add_special_tokens=False,return_offsets_mapping=True)
        ids=[tokenizer.bos_token_id]+en['input_ids'];bad=[tokenizer.bos_token_id]+ec['input_ids']
        spans=[m.span() for m in re.finditer(r'\b'+re.escape(row['subject'])+r'\b',clean)]
        assert len(spans)==2
        start,end=spans[1];hit=[i+1 for i,(a,b) in enumerate(en['offset_mapping']) if a<end and b>start];assert len(hit)==1
        s2=hit[0]
        assert len(ids)==len(bad) and [i for i,(a,b) in enumerate(zip(ids,bad)) if a!=b]==[s2]
        row.update(tokens_clean=ids,tokens_corrupt=bad,s2=s2,answer_position=len(ids)-1,
            name_index=int(row['group_id'].split(':names')[1].split(':')[0]),
            context_index=None if 'block_id' not in row else int(row['block_id'].rsplit('context',1)[1]),
            tokenization_status='qualified',token_offsets=en['offset_mapping'])
        for key,field in [('io','io_id'),('subject','subject_id'),('alternative','alt_id')]:
            one=tokenizer.encode(' '+row[key],add_special_tokens=False);assert len(one)==1;row[field]=one[0]
        assert ids[s2]==row['subject_id'] and bad[s2]==row['alt_id']
    result=dict(created_utc=c.now(),sources=sources,recipients=recipients,plans=data['interventions.jsonl'],
        original_tokenization_failures=failures,amendment='Replace corruption-only name Uma with single-token Nina before any model measurement; all affected complete matched blocks retained.',
        tokenizer_hashes={p.name:c.sha(p) for p in c.MODEL.iterdir() if p.name in ['tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']},
        bos_token_id=tokenizer.bos_token_id,all_single_token_names=True,all_corruptions_one_token=True)
    output=c.ROOT/'data/pilot-qualified.json';assert not output.exists();c.write(output,result)
    spec=c.read(c.PROJECT/'reference/source-config.json')['models']['gpt2-medium']   # release: shipped copy
    cfg=dict(seed=2026100808,device='cuda',batch_size=8,source_heads=spec['source_heads'],response_blocks=[13,18,23],
        clean_top1_minimum=.85,clean_corrupt_margin_separation_minimum=.5,positive_margin_separation_fraction_minimum=.75,
        numerical_absolute_tolerance=.0005,strengths=[-.5,.5,1.],random_replicates=4)
    c.write(c.ROOT/'protocol/config.json',cfg)
    print(json.dumps(dict(sources=len(sources),recipients=len(recipients),conditions=len(result['plans']),tokenization='passed',amendment=result['amendment'])))


if __name__=='__main__':main()
