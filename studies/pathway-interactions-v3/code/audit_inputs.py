"""Independent retokenization, bank separation, source mapping and role checks."""
import core as c


def main():
    from transformers import AutoTokenizer
    b=c.read(c.ROOT/'data/inputs.json');tok=AutoTokenizer.from_pretrained(str(c.MODEL),local_files_only=True,trust_remote_code=False)
    rows=b['recipients'];sources=b['sources'];lookup={s['id']:s for s in sources}
    for r in rows+sources:
        for field,target in [('clean_text','tokens_clean'),('corrupted_text','tokens_corrupt')]:
            assert [50256]+tok.encode(r[field],add_special_tokens=False)==r[target]
        clean,bad=r['tokens_clean'],r['tokens_corrupt'];assert len(clean)==len(bad)
        assert [i for i,(a,z) in enumerate(zip(clean,bad)) if a!=z]==[r['s2']]
        assert clean[r['s2']]==r['subject_id'] and bad[r['s2']]==r['alt_id' if r['corruption']=='third_name' else 'io_id']
        assert clean[r['io_position']]==r['io_id'] and clean[r['s1_position']]==r['subject_id']
        assert (r['io_position']<r['s1_position'])==(r['order']==0)
        assert r['answer_position']==len(clean)-1 and r['s2']<r['answer_position']
    for r in rows:
        for align,sid in r['source_ids'].items():
            s=lookup[sid]
            for k in ['name_index','role','io','subject','alternative','corruption','split']:assert s[k]==r[k]
            assert s['order']==(r['order'] if align=='matched' else 1-r['order'])
        assert lookup[r['norm_reference_id']]['order']==0
    previous=[]
    for rel in ['position-routing-v2/data/inputs.json','fixed-signal-v1/data/pilot-qualified.json','fixed-signal-v1/data/confirmation-qualified.json']:
        old=c.read(c.ROOT.parent/rel);previous+=old['recipients']+old['sources']
    fresh=[r for r in rows+sources if r['split']=='confirmation'];names={r[k] for r in fresh for k in ['io','subject','alternative']}
    assert len(names)==24 and not names&{r[k] for r in previous for k in ['io','subject','alternative']}
    assert not {r[k] for r in fresh for k in ['clean_text','corrupted_text']}&{r[k] for r in previous for k in ['clean_text','corrupted_text']}
    assert not {tuple(x) for x in b['confirmation_contexts']}&{(r['place'],r['object']) for r in previous}
    for split,n,expected in [('development',4,256),('confirmation',8,1024)]:
        rr=[r for r in rows if r['split']==split];assert len(rr)==expected
        assert {r['name_index'] for r in rr}==set(range(n)) and {r['context_index'] for r in rr}==set(range(n))
        for ni in range(n):
            for ci in range(n):assert len([r for r in rr if r['name_index']==ni and r['context_index']==ci])==16
    for study,digest in b['prior_manifests'].items():assert c.sha(c.ROOT.parent/study/'EVIDENCE-MANIFEST.json')==digest
    output=c.ROOT/'analysis/INPUT-AUDIT.json';assert not output.exists()
    c.write(output,dict(status='passed',created_utc=c.now(),retokenized_pairs=len(rows)+len(sources),recipient_cases=len(rows),source_pairs=len(sources),
        confirmation_cases=1024,confirmation_names=24,confirmation_name_units=8,confirmation_contexts=8,corruptions=2,
        fresh_templates=2,all_role_order_and_source_mappings_verified=True,prior_manifests_unchanged=True,source_sha256=c.sha(__file__)))
    print('Input audit passed',flush=True)


if __name__=='__main__':main()
