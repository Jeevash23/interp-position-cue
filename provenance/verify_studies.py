"""Run the three frozen study verifiers unchanged, against the relocated preparation files (make verify).

Why this wrapper exists. Each study's code/verify_study.py checks, by path, 25 preparation files at the project root
(README.md, Makefile, .gitignore, 00-START-HERE.md and the folders 01-Papers to 06-Study-History, all from 7 October
2026) and four files of the companion project AI-01 under the field root. On 9 October 2026:
  * the 25 preparation files were moved, byte for byte, to provenance/preparation-2026-10-07/ so that the project root
    could carry current entry pages (record: provenance/relocations-2026-10-09.json);
  * AI-01 revised its REPRODUCIBILITY.md; the version the studies recorded is preserved byte for byte in
    provenance/upstream-references/ (and in AI-01's provenance/document-revisions/2026-10-09-v3.1/).
The frozen verifiers cannot be edited, so this wrapper:
  1. checks every entry of the relocation record (new path, size and SHA-256; for preparation files, the hash recorded
     in Study 1's provenance/PREPARATION-SNAPSHOT.json);
  2. checks each AI-01 reference file, using the live file while it still has its recorded hash and the preserved copy
     otherwise (reported as such);
  3. runs each study's own verify_study.py in a fresh interpreter, with core.PROJECT pointed at the preparation files
     and core.FIELD at a temporary view that holds the four reference files. Study 3's verifier calls Study 2's in a
     subprocess; that call is routed through the same mechanism. All other checks of the frozen verifiers (manifests,
     freeze records, model assets, accounting and decisions) run exactly as written.

Usage:
  python -B provenance/verify_studies.py                      # verify all three studies
  python -B provenance/verify_studies.py run SCRIPT [ARGS...] # run any frozen study script under the same redirection
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THIS = Path(__file__).resolve()
PROJECT = THIS.parents[1]
FIELD = PROJECT.parent
STUDIES = ['fixed-signal-v1', 'position-routing-v2', 'pathway-interactions-v3']
RECORD = PROJECT / 'provenance/relocations-2026-10-09.json'
PREP_SNAPSHOT = PROJECT / 'studies/fixed-signal-v1/provenance/PREPARATION-SNAPSHOT.json'
PRESERVED = PROJECT / 'provenance/upstream-references'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            h.update(block)
    return h.hexdigest()


def preparation_root():
    """Directory that holds the 25 preparation files at their original relative paths."""
    if RECORD.exists():
        return PROJECT / json.loads(RECORD.read_text())['preparation_root']
    return PROJECT


def reference_sources():
    """For each AI-01 reference file: (relative path, file to use, 'live' or 'preserved copy')."""
    prep = json.loads(PREP_SNAPSHOT.read_text())
    out = []
    for item in prep['reference_files']:
        rel = item['path_from_field_root']
        live, kept = FIELD / rel, PRESERVED / rel
        if live.is_file() and sha(live) == item['sha256']:
            out.append((rel, live, 'live'))
        elif kept.is_file() and sha(kept) == item['sha256']:
            out.append((rel, kept, 'preserved copy'))
        else:
            raise SystemExit(f'reference file {rel}: neither the live file nor the preserved copy has the recorded hash')
    return out


def field_view():
    view = Path(tempfile.mkdtemp(prefix='ai08-field-view-'))
    atexit.register(shutil.rmtree, view, True)
    for rel, source, _ in reference_sources():
        target = view / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source)
    return view


def run(script, args):
    """Run one frozen study script with core.PROJECT and core.FIELD redirected (see module docstring)."""
    script = Path(script).resolve()
    if script.parent.parent.parent != PROJECT / 'studies' or script.parent.name != 'code':
        raise SystemExit(f'{script} is not a study script under studies/*/code/')
    sys.path.insert(0, str(script.parent))
    import core                                                   # the study's own core module
    assert Path(core.PROJECT).resolve() == PROJECT, core.PROJECT
    core.PROJECT = preparation_root()
    core.FIELD = field_view()                                     # core.MODEL was fixed at import from the real field
    original = subprocess.run

    def routed(cmd, *a, **kw):
        if isinstance(cmd, (list, tuple)) and cmd and str(cmd[-1]).endswith('.py') and \
                Path(str(cmd[-1])).resolve().parent.parent.parent == PROJECT / 'studies':
            cmd = [sys.executable, '-B', str(THIS), 'run', str(cmd[-1])]
        return original(cmd, *a, **kw)

    subprocess.run = routed
    sys.argv = [str(script)] + list(args)
    runpy.run_path(str(script), run_name='__main__')


def check_relocations():
    if not RECORD.exists():
        print('relocation record absent: preparation files are read from the project root')
        return 0
    record = json.loads(RECORD.read_text())
    prep = json.loads(PREP_SNAPSHOT.read_text())['files']
    bad = 0
    for item in record['moves']:
        new = PROJECT / item['to']
        ok = new.is_file() and new.stat().st_size == item['bytes'] and sha(new) == item['sha256']
        if item['from'] in prep:
            ok = ok and prep[item['from']] == item['sha256']
        if not ok:
            bad += 1
            print(f"FAIL relocation {item['from']} -> {item['to']}")
    root = preparation_root()
    missing = [rel for rel in prep if not (root / rel).is_file()]
    if missing:
        bad += len(missing)
        print('FAIL preparation files missing from', root.relative_to(PROJECT), missing)
    print(f"ok  {len(record['moves']) - bad} of {len(record['moves'])} relocated files match the record; "
          f"preparation files read from {root.relative_to(PROJECT)}/")
    return bad


def main():
    bad = check_relocations()
    for rel, source, how in reference_sources():
        print(f'ok  AI-01 reference {rel}: {how}')
    results = {}
    for study in STUDIES:
        script = PROJECT / 'studies' / study / 'code/verify_study.py'
        r = subprocess.run([sys.executable, '-B', str(THIS), 'run', str(script)], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        try:
            out = json.loads(r.stdout)
        except ValueError:
            out = {'status': 'failed', 'stderr': r.stderr.strip().splitlines()[-3:]}
        results[study] = out
        status = out.get('status')
        print(f"{'ok ' if status == 'passed' else 'FAIL'} {study}/code/verify_study.py: {status}"
              + (f" ({out.get('verified_study_files', (out.get('causal_followup') or {}).get('files'))} files)" if status == 'passed' else f" {out.get('stderr')}"))
        bad += status != 'passed'
    if bad:
        sys.exit(f'{bad} verification problem(s)')
    print('all three study verifiers passed')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'run':
        run(sys.argv[2], sys.argv[3:])
    else:
        main()
