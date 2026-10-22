"""Independent token, semantic-position and donor/recipient separation checks."""
from pathlib import Path
import numpy as np
import core as c


def main():
    from transformers import AutoTokenizer
    output=c.ROOT/'analysis/INPUT-AUDIT.json';assert not output.exists();b=c.read(c.ROOT/'data/inputs.json')
    tok=AutoTokenizer.from_pretrained(str(c.MODEL),local_files_only=True,trust_remote_code=False)
    prior=c.ROOT.parent/'fixed-signal-v1';old=[c.read(prior/f'data/{name}-qualified.json') for name in ['pilot','confirmation']]
    oldnames={r[k] for d in old for r in d['sources'] for k in ['io','subject','alternative']}
    freshunits=b['confirmation_units'];names=[x for u in freshunits for group in u.values() for x in group]
    assert len(names)==48 and len(set(names))==48 and not set(names)&oldnames
    assert not {(r['place'],r['object']) for d in old for r in d['recipients']} & {tuple(v) for v in b['confirmation_contexts']}
    source_map={r['id']:r for r in b['sources']};count=0
    for row in b['recipients']+b['sources']:
        count+=1
        for txt,tokens in [('clean_text','tokens_clean'),('corrupted_text','tokens_corrupt')]:assert [50256]+tok.encode(row[txt],add_special_tokens=False)==row[tokens]
        tc=np.array(row['tokens_clean']);tb=np.array(row['tokens_corrupt']);assert np.where(tc!=tb)[0].tolist()==[row['s2']]
        assert tc[row['io_position']]==row['io_id'] and tc[row['s1_position']]==row['subject_id'] and tc[row['s2']]==row['subject_id'] and tb[row['s2']]==row['alt_id']
        for key,idkey in [('io','io_id'),('subject','subject_id'),('alternative','alt_id')]:assert tok.encode(' '+row[key],add_special_tokens=False)==[row[idkey]]
        if row['context_index'] is not None:
            for label,sid in row['source_ids'].items():
                source=source_map[sid];kind,alignment=label.split('_')
                assert source['name_index']==row['name_index'] and source['role']==row['role'] and source['split']==row['split']
                assert (source['order']==row['order'])==(alignment=='matched')
                identities={row[k] for k in ['io','subject','alternative']};donor={source[k] for k in ['io','subject','alternative']}
                assert (identities==donor) if kind=='same' else not identities&donor
    cfg=c.read(c.ROOT/'protocol/config.json')
    for controls in b['control_head_sets']:
        assert len(controls)==5 and [l for l,h in controls]==[l for l,h in cfg['consumer_heads']]
        assert not set(map(tuple,controls))&set(map(tuple,cfg['consumer_heads']))
    for freeze in (c.ROOT/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze)['files'].items():assert c.sha(c.ROOT/rel)==digest
    c.write(output,dict(status='passed',created_utc=c.now(),retokenized_pairs=count,confirmation_names=48,confirmation_name_units=8,
        confirmation_contexts=8,sources=len(b['sources']),recipients=len(b['recipients']),new_names_and_context_pairs=True,
        donor_recipient_dependencies_contained_within_name_unit=True,alignment_and_role_checks=True,source_sha256=c.sha(__file__)))
    print('Input audit passed on',count,'pairs',flush=True)


if __name__=='__main__':main()
