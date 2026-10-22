import unittest
import numpy as np
import response_models as m


class ModelChecks(unittest.TestCase):
    def test_gain_and_rotation(self):
        rng=np.random.default_rng(73001);x=np.zeros((120,12));x[:,:2]=rng.normal(size=(120,2));keys=['a']*120;cells=['c']*120
        gain=m.fit(x,2*x,keys,cells);np.testing.assert_allclose(m.predict(gain,x,keys,cells),2*x,atol=1e-12)
        y=np.zeros_like(x);y[:,0]=-x[:,1];y[:,1]=x[:,0]
        base=m.fit(x,y,keys,cells);rich=m.fit(x,y,keys,cells,2,1e-9)
        p=m.predict(rich,x,keys,cells);self.assertLess(np.sum((y-p)**2)/np.sum(y*y),1e-12)
        self.assertGreater(np.sum((y-m.predict(base,x,keys,cells))**2)/np.sum(y*y),.99)
        total,perp=m.errors(x,y,m.predict(base,x,keys,cells));np.testing.assert_allclose(total,perp,atol=1e-12)

    def test_crossed_separation_and_aggregation(self):
        names=np.repeat(np.arange(4),4);contexts=np.tile(np.arange(4),4)
        for n,c,train,test in m.folds(names,contexts):
            self.assertFalse(np.any(names[train]==n));self.assertFalse(np.any(contexts[train]==c));self.assertEqual(int(test.sum()),1)
        interval=m.crossed_interval(np.ones(16)*2,np.ones(16)*4,names,contexts,draws=50)
        self.assertEqual(interval['mean'],.5);self.assertEqual(interval['ci95'],[.5,.5])


if __name__=='__main__':unittest.main(verbosity=2)
