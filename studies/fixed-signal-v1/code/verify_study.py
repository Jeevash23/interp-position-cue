"""Read-only verification of the completed study's file manifest and closure accounting."""
import json
import core as c


def main():
    manifest=c.read(c.ROOT/'EVIDENCE-MANIFEST.json');failures=[]
    for rel,item in manifest['files'].items():
        p=c.ROOT/rel
        if not p.is_file() or p.stat().st_size!=item['bytes'] or c.sha(p)!=item['sha256']:failures.append(rel)
    current={str(p.relative_to(c.ROOT)) for p in c.ROOT.rglob('*') if p.is_file() and p.name!='EVIDENCE-MANIFEST.json'}
    extra=sorted(current-set(manifest['files']));missing=sorted(set(manifest['files'])-current)
    assert not failures and not extra and not missing,dict(changed=failures,extra=extra,missing=missing)
    for freeze in (c.ROOT/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze).get('files',{}).items():assert c.sha(c.ROOT/rel)==digest,(str(freeze),rel)
    prep=c.read(c.ROOT/'provenance/PREPARATION-SNAPSHOT.json')
    for rel,digest in prep['files'].items():assert c.sha(c.PROJECT/rel)==digest
    for item in prep['reference_files']:assert c.sha(c.FIELD/item['path_from_field_root'])==item['sha256']
    model=c.read(c.ROOT/'provenance/MODEL-ASSETS.json')
    for rel,item in model['files'].items():assert c.sha(c.MODEL/rel)==item['sha256']
    summary=c.read(c.ROOT/'analysis/STUDY-SUMMARY.json')
    assert summary['total_condition_records']==18304 and summary['model_examples']['total']==25544
    assert summary['decision']['controlled_program_complete'] and not summary['decision']['routing_mechanism_identified']
    assert c.read(c.ROOT/'runs/model-replay/summary.json')['status']=='failed'
    assert c.read(c.ROOT/'runs/batch-replay/summary.json')['bitwise_identical']
    print(json.dumps(dict(status='passed',verified_study_files=len(manifest['files']),bytes=manifest['total_bytes'],
        upstream_and_frozen_files_preserved=True,condition_records=18304,model_examples=25544,
        scope='Evidence integrity and accurate closure, including the documented singleton tolerance failure'),indent=2))


if __name__=='__main__':main()
