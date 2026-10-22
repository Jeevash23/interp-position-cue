"""Rerun each study's frozen saved-output analysis into a temporary folder and compare it with the saved analysis.

The commands are those documented in each study's REPRODUCIBILITY.md (Study 3 with its serialization wrapper). Each
study runs in its own interpreter, because each has its own `core` module. JSON files must be equal as values and
NPZ files array by array (container timestamps are ignored). Nothing under studies/ is written; the temporary folders
are deleted afterwards. Writes analysis/results/replay-check.json. No model and no GPU are used.
Run: python -B analysis/replay_study_analyses.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'analysis/results/replay-check.json'
STUDIES = {
    'fixed-signal-v1': ('from analyze_responses import analyze\n'
                        "analyze(study / 'runs/development', output / 'development')\n"
                        "analyze(study / 'runs/holdout', output / 'holdout', output / 'development')\n"
                        "analyze(study / 'runs/confirmation', output / 'confirmation', output / 'development')\n",
                        ['development', 'holdout', 'confirmation']),
    'position-routing-v2': ('from analyze_causal import analyze\n'
                            "for bank in ['development', 'confirmation']:\n    analyze(study / 'runs' / bank, output / bank)\n",
                            ['development', 'confirmation']),
    'pathway-interactions-v3': ('import core\nfrom serialize_results import native\nfrom analyze import analyze\n'
                                'original_write = core.write\n'
                                'core.write = lambda path, value: original_write(path, native(value))\n'
                                "for bank in ['development', 'confirmation']:\n    analyze(study / 'runs' / bank, output / bank)\n",
                                ['development', 'confirmation']),
}
HEAD = 'import sys\nfrom pathlib import Path\nstudy = Path(sys.argv[1]); output = Path(sys.argv[2])\n' \
       "sys.path.insert(0, str(study / 'code'))\n"


def same_file(saved, replay):
    if saved.suffix == '.json':
        return json.loads(saved.read_text()) == json.loads(replay.read_text())
    if saved.suffix == '.npz':
        with np.load(saved) as a, np.load(replay) as b:
            return set(a.files) == set(b.files) and all(
                a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k], equal_nan=True)
                for k in a.files)
    return saved.read_bytes() == replay.read_bytes()


def main():
    report, failed = {}, 0
    for study, (code, banks) in STUDIES.items():
        root = PROJECT / 'studies' / study
        tmp = Path(tempfile.mkdtemp(prefix=f'ai08-replay-{study}-'))
        try:
            r = subprocess.run([sys.executable, '-B', '-c', HEAD + code, str(root), str(tmp)], capture_output=True, text=True)
            if r.returncode:
                report[study] = dict(status='failed', error=r.stderr.strip().splitlines()[-1:])
                failed += 1
                continue
            files = {}
            for bank in banks:
                saved_dir, replay_dir = root / 'analysis' / bank, tmp / bank
                names = sorted(p.name for p in saved_dir.iterdir() if p.is_file())
                extra = sorted({p.name for p in replay_dir.iterdir() if p.is_file()} - set(names))
                for name in names:
                    files[f'{bank}/{name}'] = (replay_dir / name).is_file() and same_file(saved_dir / name, replay_dir / name)
                for name in extra:
                    files[f'{bank}/{name}'] = False
            ok = all(files.values())
            failed += not ok
            report[study] = dict(status='passed' if ok else 'failed', files_equal=sum(files.values()), files=len(files),
                                 unequal=[k for k, v in files.items() if not v])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f"{'ok ' if report[study]['status'] == 'passed' else 'FAIL'} {study}: "
              f"{report[study].get('files_equal', 0)} of {report[study].get('files', '?')} analysis files reproduced exactly")
    import scipy
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(checked_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                   python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                                   studies=report), indent=1) + '\n')
    if failed:
        sys.exit(f'{failed} study analysis replay(s) differ')


if __name__ == '__main__':
    main()
