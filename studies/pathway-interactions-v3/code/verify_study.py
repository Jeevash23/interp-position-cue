"""Read-only evidence verification, including the unchanged two earlier AI-08 studies."""
import json,subprocess,sys
import core as c


def main():
    root=c.ROOT;manifest=c.read(root/'EVIDENCE-MANIFEST.json')
    files={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and p!=root/'EVIDENCE-MANIFEST.json'}
    assert files==set(manifest['files']),dict(extra=sorted(files-set(manifest['files'])),missing=sorted(set(manifest['files'])-files))
    for rel,item in manifest['files'].items():
        path=root/rel;assert path.stat().st_size==item['bytes'] and c.sha(path)==item['sha256'],rel
    assert manifest['total_bytes']==sum(v['bytes'] for v in manifest['files'].values())
    for freeze in (root/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze)['files'].items():assert c.sha(root/rel)==digest,(str(freeze),rel)
    for name in ['INPUT-AUDIT','EVIDENCE-AUDIT','ANALYSIS-REPLAY','SERIALIZATION-EQUALITY']:
        assert c.read(root/f'analysis/{name}.json')['status']=='passed'
    s=c.read(root/'analysis/STUDY-SUMMARY.json');confirmation=c.read(root/'analysis/confirmation/summary.json')
    assert s['status']=='complete' and s['decision']['planned_followup_complete']
    assert s['decision']['strong_explanation_supported']==all(confirmation['gates'].values())
    assert not s['decision']['complete_mechanism_identified']
    assert s['model_examples']['total']==158336 and s['total_condition_records']==128448
    assert s['unique_recipient_cases']==1280 and s['recipient_panel_evaluations']==1312
    assert c.read(root/'runs/replay/summary.json')['bitwise_identical']
    bundle=c.read(root/'data/inputs.json')
    for name,digest in bundle['prior_manifests'].items():assert c.sha(root.parent/name/'EVIDENCE-MANIFEST.json')==digest
    prior=subprocess.run([sys.executable,'-B',str(root.parent/'position-routing-v2/code/verify_study.py')],check=True,capture_output=True,text=True)
    old=json.loads(prior.stdout);assert old['status']=='passed'
    print(json.dumps(dict(status='passed',verified_study_files=len(manifest['files']),bytes=manifest['total_bytes'],
        prior_studies_verified=old,model_examples=158336,condition_records=128448,strong_explanation_supported=s['decision']['strong_explanation_supported'],
        scope='Integrity, complete accounting and preserved failure/repair records; scientific gates are reported without relaxation.'),indent=2))


if __name__=='__main__':main()
