"""Check crossed-unit inference against fixed synthetic ground truth."""
import unittest
import numpy as np
from analyze import crossed


class StatisticsTests(unittest.TestCase):
    def test_proportional_route_ratio_survives_heterogeneous_denominators(self):
        rows=[dict(name_index=n,context_index=k) for n in range(4) for k in range(5)]
        d=np.array([.5+2*r['name_index']+.3*r['context_index'] for r in rows])
        _,boot,_=crossed(dict(total=d,part=.37*d),rows,2000,81)
        np.testing.assert_allclose(boot['part']/boot['total'],.37,rtol=1e-14)

    def test_within_unit_repetition_cannot_narrow_uncertainty(self):
        rows=[dict(name_index=n,context_index=k) for n in range(4) for k in range(5)]
        v=np.array([2*r['name_index']-r['context_index'] for r in rows],float)
        a,b,_=crossed(dict(effect=v),rows,2000,82)
        repeated=[r for r in rows for _ in range(8)]
        aa,bb,_=crossed(dict(effect=np.repeat(v,8)),repeated,2000,82)
        self.assertEqual(a,aa);np.testing.assert_array_equal(b['effect'],bb['effect'])


if __name__=='__main__':unittest.main()
