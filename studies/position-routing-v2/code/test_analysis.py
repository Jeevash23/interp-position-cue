"""Analytical tests for preservation of paired groups in the causal estimator."""
import unittest
import numpy as np
from analyze_causal import intervals


class CrossedInferenceTests(unittest.TestCase):
    def test_constant_ratio_with_heterogeneous_denominators(self):
        rows=[dict(name_index=n,context_index=c) for n in range(3) for c in range(4) for _ in range(8)]
        denominator=np.array([1+r['name_index']+2*r['context_index'] for r in rows],float)
        _,boot=intervals(dict(d=denominator,n=.6*denominator),rows,draws=500)
        self.assertTrue(np.allclose(boot['n']/boot['d'],.6,atol=1e-14))

    def test_within_unit_repetition_does_not_inflate_precision(self):
        rows=[dict(name_index=n,context_index=c) for n in range(3) for c in range(4)]
        values=np.array([r['name_index']-2*r['context_index'] for r in rows],float)
        s1,b1=intervals(dict(effect=values),rows,draws=500)
        rr=[r for r in rows for _ in range(8)];vv=np.repeat(values,8)
        s2,b2=intervals(dict(effect=vv),rr,draws=500)
        self.assertEqual(s1,s2);self.assertTrue(np.array_equal(b1['effect'],b2['effect']))


if __name__=='__main__':unittest.main()
