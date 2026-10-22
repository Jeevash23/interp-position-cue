"""Seal completed local evidence once, without altering any frozen scientific files."""
import ast
import core as c


def main():
    output = c.ROOT / 'EVIDENCE-MANIFEST.json'
    assert not output.exists()
    assert c.read(c.ROOT / 'analysis/STUDY-SUMMARY.json')['status'] == 'complete'
    for name in ['INPUT-AUDIT', 'EVIDENCE-AUDIT', 'ANALYSIS-REPLAY']:
        assert c.read(c.ROOT / f'analysis/{name}.json')['status'] == 'passed'
    for freeze in (c.ROOT / 'protocol').glob('*FREEZE.json'):
        for rel, digest in c.read(freeze)['files'].items():
            assert c.sha(c.ROOT / rel) == digest, rel
    for path in (c.ROOT / 'code').glob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    files = {}
    for path in sorted(c.ROOT.rglob('*')):
        if path.is_file():
            assert '__pycache__' not in path.parts and path.suffix != '.partial', str(path)
            files[str(path.relative_to(c.ROOT))] = dict(bytes=path.stat().st_size, sha256=c.sha(path))
    total = sum(item['bytes'] for item in files.values())
    c.write(output, dict(created_utc=c.now(), study='position-routing-v2', files=files,
                         total_bytes=total, scope='All completed causal-follow-up evidence except this manifest itself.'))
    print('Sealed', len(files), 'files;', total, 'bytes')


if __name__ == '__main__':
    main()
