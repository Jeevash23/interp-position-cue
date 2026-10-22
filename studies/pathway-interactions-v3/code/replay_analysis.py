"""Regenerate the completed factorial analyses in a fresh temporary directory."""
import contextlib,tempfile
from pathlib import Path
import numpy as np
import core as c
from analyze import analyze


def main():
    output=c.ROOT/'analysis/ANALYSIS-REPLAY.json';assert not output.exists();tmp=Path(tempfile.mkdtemp(prefix='ai08-v3-analysis-',dir='/tmp'));checks=[]
    for phase in ['development','confirmation']:
        with (tmp/f'{phase}.log').open('w') as f,contextlib.redirect_stdout(f):analyze(c.ROOT/'runs'/phase,tmp/phase)
        assert c.read(c.ROOT/f'analysis/{phase}/summary.json')==c.read(tmp/phase/'summary.json')
        with np.load(c.ROOT/f'analysis/{phase}/paired-scores.npz') as a,np.load(tmp/phase/'paired-scores.npz') as b:
            assert a.files==b.files
            for k in a.files:assert np.array_equal(a[k],b[k]),(phase,k)
        checks.append(phase);print(phase,'analysis exactly reproduced',flush=True)
    c.write(output,dict(status='passed',banks=checks,temporary_directory=str(tmp),source_sha256=c.sha(__file__),created_utc=c.now()))


if __name__=='__main__':main()
