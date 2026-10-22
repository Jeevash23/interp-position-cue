"""Regenerate both saved-data causal analyses into a fresh temporary directory."""
import contextlib
import tempfile
from pathlib import Path
import numpy as np
import core as c
from analyze_causal import analyze


def main():
    output=c.ROOT/'analysis/ANALYSIS-REPLAY.json';assert not output.exists();tmp=Path(tempfile.mkdtemp(prefix='ai08-causal-replay-',dir='/tmp'));files=[]
    for bank in ['development','confirmation']:
        with (tmp/f'{bank}.log').open('w') as f,contextlib.redirect_stdout(f):analyze(c.ROOT/'runs'/bank,tmp/bank)
        for original in sorted((c.ROOT/'analysis'/bank).iterdir()):
            if original.suffix=='.json':assert c.read(original)==c.read(tmp/bank/original.name)
            elif original.suffix=='.npz':
                with np.load(original) as a,np.load(tmp/bank/original.name) as b:
                    assert a.files==b.files
                    for k in a.files:assert np.array_equal(a[k],b[k])
            else:continue
            files.append(f'{bank}/{original.name}')
        print(bank,'analysis reproduced exactly',flush=True)
    c.write(output,dict(status='passed',created_utc=c.now(),files=files,temporary_directory=str(tmp),source_sha256=c.sha(__file__)))


if __name__=='__main__':main()
