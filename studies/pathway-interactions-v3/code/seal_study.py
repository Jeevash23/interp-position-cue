"""Seal completed evidence once after audits and closure accounting."""
import ast
import core as c


def main():
    output=c.ROOT/'EVIDENCE-MANIFEST.json';assert not output.exists()
    assert c.read(c.ROOT/'analysis/STUDY-SUMMARY.json')['status']=='complete'
    for freeze in (c.ROOT/'protocol').glob('*FREEZE.json'):
        for rel,digest in c.read(freeze)['files'].items():assert c.sha(c.ROOT/rel)==digest,(str(freeze),rel)
    for p in (c.ROOT/'code').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    files={}
    for p in sorted(c.ROOT.rglob('*')):
        if not p.is_file():continue
        assert '__pycache__' not in p.parts and p.suffix!='.partial',str(p)
        files[str(p.relative_to(c.ROOT))]=dict(bytes=p.stat().st_size,sha256=c.sha(p))
    c.write(output,dict(created_utc=c.now(),study='pathway-interactions-v3',files=files,total_bytes=sum(v['bytes'] for v in files.values()),
        scope='All retained study evidence, including the preserved serialization failure, except this manifest itself.'))
    print('Sealed',len(files),'files;',sum(v['bytes'] for v in files.values()),'bytes')


if __name__=='__main__':main()
