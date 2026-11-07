import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import group_loss_experiment as experiment
from group_loss_experiment import GroupObjective


class GroupLossContracts(unittest.TestCase):
    def test_gradient_matches_independent_finite_difference(self):
        c=np.array([100.,200.,100.,300.]);codes=np.array([0,0,1,1])
        w=c*np.array([2.,2.,3.,3.]);y=np.array([.1,.2,.3,.4]);pred=np.array([.15,.1,.35,.5])
        obj=GroupObjective(y,c,w,codes,np.ones(4))
        grad,hess=obj(pred)
        eps=1e-6
        numeric=[]
        for i in range(4):
            step=np.eye(4)[i]*eps
            numeric.append((obj.value(pred+step)-obj.value(pred-step))/(2*eps))
        np.testing.assert_allclose(grad,numeric,rtol=1e-6,atol=1e-7)
        self.assertTrue((hess>0).all())
        matrix=np.column_stack([(obj(pred+np.eye(4)[i]*eps)[0]-
                                 obj(pred-np.eye(4)[i]*eps)[0])/(2*eps) for i in range(4)])
        self.assertGreaterEqual(np.linalg.eigvalsh(np.diag(hess)-matrix).min(),-1e-6)

    def test_group_cancellation_and_partial_capacity_fallback(self):
        obj=GroupObjective([0,0],[1,1],[1,1],[0,0],[1,1],mix=1)
        grad,_=obj(np.array([1.,-1.]))
        np.testing.assert_array_equal(grad,[0,0])
        plain=GroupObjective([0,0],[1,1],[1,1],[0,0],[0,0],mix=1)
        grad,_=plain(np.array([1.,-1.]))
        np.testing.assert_array_equal(grad,[1,-1])

    def test_unselected_model_has_no_invented_standalone_score(self):
        grouped=pd.DataFrame(dict(ts=pd.to_datetime(['2025-03-01T12:00+09:00',
            '2025-04-01T12:00+09:00','2025-05-01T12:00+09:00']),
            base=[11.,11.,11.],energy=[10.,10.,10.],weight=[100.,100.,100.]))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(experiment,'select_model',return_value={'new_weight':0}), \
                 patch.object(experiment.pd,'read_parquet',return_value=pd.DataFrame({'ensemble':[11.]})), \
                 patch.object(experiment.ex,'group_frame',return_value=grouped), \
                 patch.object(experiment.p,'log'):
                experiment.evaluate(root,run=root)
            result=experiment.ex.read(root/'evaluation.json')
            self.assertFalse(result['accepted'])
            self.assertFalse(result['standalone_development_evaluated'])
            self.assertTrue(all(r['new_model_only_penalty_krw'] is None for r in result['months']))
            self.assertTrue(all(r['gain_pct']==0 for r in result['months']))

    def test_selected_model_requires_real_development_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with patch.object(experiment,'select_model',return_value={'new_weight':.25}), \
                 patch.object(experiment.pd,'read_parquet',return_value=pd.DataFrame({'ensemble':[11.]})):
                with self.assertRaisesRegex(ValueError,'requires development predictions'):
                    experiment.evaluate(root,run=root)


if __name__=='__main__':unittest.main()
