"""Read-only verification of both AI-08 evidence seals and the honest V2 conclusion."""
import json
from pathlib import Path
import core as c


def verify_manifest(root):
    manifest = c.read(root / 'EVIDENCE-MANIFEST.json')
    current = {str(p.relative_to(root)) for p in root.rglob('*')
               if p.is_file() and p.name != 'EVIDENCE-MANIFEST.json'}
    assert current == set(manifest['files']), dict(extra=sorted(current - set(manifest['files'])),
                                                   missing=sorted(set(manifest['files']) - current))
    for rel, item in manifest['files'].items():
        path = root / rel
        assert path.stat().st_size == item['bytes'] and c.sha(path) == item['sha256'], str(path)
    assert sum(item['bytes'] for item in manifest['files'].values()) == manifest['total_bytes']
    for freeze in (root / 'protocol').glob('*FREEZE.json'):
        for rel, digest in c.read(freeze)['files'].items():
            assert c.sha(root / rel) == digest, (str(freeze), rel)
    return dict(files=len(manifest['files']), bytes=manifest['total_bytes'])


def main():
    current = verify_manifest(c.ROOT)
    previous_ref = c.read(c.ROOT / 'provenance/PREVIOUS-STUDY.json')
    previous = c.ROOT.parent / 'fixed-signal-v1'
    assert c.sha(previous / 'EVIDENCE-MANIFEST.json') == previous_ref['sha256']
    prior = verify_manifest(previous)
    prep = c.read(previous / 'provenance/PREPARATION-SNAPSHOT.json')
    for rel, digest in prep['files'].items():
        assert c.sha(c.PROJECT / rel) == digest
    for item in prep['reference_files']:
        assert c.sha(c.FIELD / item['path_from_field_root']) == item['sha256']
    assets = c.read(c.ROOT / 'provenance/MODEL-ASSETS.json')
    assert assets == c.read(previous / 'provenance/MODEL-ASSETS.json')
    for rel, item in assets['files'].items():
        assert (c.MODEL / rel).stat().st_size == item['bytes'] and c.sha(c.MODEL / rel) == item['sha256']
    s = c.read(c.ROOT / 'analysis/STUDY-SUMMARY.json')
    assert s['status'] == 'complete' and s['decision']['planned_followup_complete']
    assert not s['decision']['strong_joint_criterion_passed']
    assert s['decision']['partial_causal_evidence'] and not s['decision']['complete_mechanism_identified']
    assert s['recipients'] == 640 and s['sources'] == 96 and s['total_condition_records'] == 28800
    assert s['model_examples']['total'] == 40256
    assert c.read(c.ROOT / 'runs/replay/summary.json')['bitwise_identical']
    for name in ['INPUT-AUDIT', 'EVIDENCE-AUDIT', 'ANALYSIS-REPLAY']:
        assert c.read(c.ROOT / f'analysis/{name}.json')['status'] == 'passed'
    print(json.dumps(dict(status='passed', causal_followup=current, prior_study=prior,
                          original_preparation_and_model_assets_preserved=True,
                          condition_records=28800, model_examples=40256,
                          scope='Integrity and completed accounting; the strong scientific explanation criterion remains failed.'), indent=2))


if __name__ == '__main__':
    main()
