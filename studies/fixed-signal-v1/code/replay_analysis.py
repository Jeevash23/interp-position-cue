"""Rerun every saved-data analysis, including development selection, in a fresh temporary folder."""
import contextlib
import tempfile
from pathlib import Path
import numpy as np
import core as c
from analyze_responses import analyze


def main():
    result=c.ROOT/'analysis/ANALYSIS-REPLAY.json';assert not result.exists()
    temporary=Path(tempfile.mkdtemp(prefix='ai08-analysis-replay-',dir='/tmp'));records=[]
    for bank in ['development','holdout','confirmation']:
        with (temporary/f'{bank}.log').open('w') as log,contextlib.redirect_stdout(log):
            analyze(c.ROOT/'runs'/bank,temporary/bank,None if bank=='development' else temporary/'development')
        original=c.ROOT/'analysis'/bank
        for p in sorted(original.iterdir()):
            if p.suffix=='.json':
                assert c.read(p)==c.read(temporary/bank/p.name),str(p)
                records.append(dict(bank=bank,file=p.name,equality='exact JSON values'))
            elif p.suffix=='.npz':
                with np.load(p) as a,np.load(temporary/bank/p.name) as b:
                    assert a.files==b.files
                    for key in a.files:assert np.array_equal(a[key],b[key]),(str(p),key)
                records.append(dict(bank=bank,file=p.name,equality='bitwise numerical arrays'))
        print(bank,'analysis replay identical',flush=True)
    c.write(result,dict(status='passed',created_utc=c.now(),temporary_path=str(temporary),files=records,
        development_selection_repeated=True,independent_holdouts_use_replayed_development_model=True,source_sha256=c.sha(__file__)))


if __name__=='__main__':main()
