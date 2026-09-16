"""Numerical invariants and saved-artifact checks without estimator libraries."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from regression import hypothesis, cost, fit_normal, fit_batch_gd, fit_sgd, rmse
from train_eval import standardize, split_by_date, FEATURES_A, FEATURES_B

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.,-1.],[1.,0.],[1.,1.],[1.,2.]])
        self.theta = np.array([2.,3.])
        self.y = self.X @ self.theta

    def test_exact_normal_equation(self):
        np.testing.assert_allclose(fit_normal(self.X,self.y), self.theta, atol=1e-12)

    def test_half_sum_cost_and_rmse(self):
        self.assertAlmostEqual(cost(self.X,self.y,np.zeros(2)),.5*np.sum(self.y**2))
        self.assertAlmostEqual(rmse(self.y,np.zeros(4)),np.sqrt(np.mean(self.y**2)))

    def test_batch_first_update_has_no_mean_factor(self):
        theta, history = fit_batch_gd(self.X,self.y,.01,1)
        np.testing.assert_allclose(theta,.01*self.X.T@self.y)
        self.assertAlmostEqual(history[0],cost(self.X,self.y,theta))

    def test_ordered_sgd_first_epoch(self):
        manual=np.zeros(2)
        for x,y in zip(self.X,self.y): manual += .01*(y-x@manual)*x
        actual,history=fit_sgd(self.X,self.y,.01,1)
        np.testing.assert_allclose(actual,manual)
        self.assertAlmostEqual(history[0],cost(self.X,self.y,actual))

    def test_column_target_cannot_broadcast(self):
        np.testing.assert_allclose(fit_normal(self.X,self.y[:,None]),self.theta)
        self.assertEqual(cost(self.X,self.y[:,None],self.theta),0)
        self.assertEqual(rmse(self.y[:,None],self.y),0)

    def test_batch_converges_and_descends(self):
        theta,history=fit_batch_gd(self.X,self.y,.05,1000)
        np.testing.assert_allclose(theta,self.theta,atol=1e-10)
        self.assertTrue(np.all(np.diff(history)<=1e-10))

    def test_invalid_data_fails(self):
        for x,y in [(np.ones((3,2)),np.ones(4)),(np.array([[1.,np.nan]]),np.ones(1))]:
            with self.assertRaises(ValueError): fit_normal(x,y)
        with self.assertRaises(ValueError): fit_normal(np.ones((4,2)),self.y)
        with self.assertRaises(ValueError): fit_batch_gd(self.X,self.y,0,5)
        with self.assertRaises(ValueError): rmse([1,2],[1])

    def test_test_rows_do_not_change_scaler(self):
        train=pd.DataFrame({'x':[1.,2.,3.]})
        test=pd.DataFrame({'x':[10000.]})
        x1,x2,means,stds=standardize(train,test,['x'])
        np.testing.assert_allclose(means,[2])
        np.testing.assert_allclose(stds,[np.std([1.,2.,3.])])
        np.testing.assert_allclose(x1[:,0],1)
        self.assertGreater(x2[0,1],1000)

    def test_saved_metrics_recompute_and_coefficients_agree(self):
        results=ROOT/'results'
        predictions=pd.read_csv(results/'test_predictions.csv')
        metrics=pd.read_csv(results/'rmse_table.csv')
        mask=predictions.irradiation.to_numpy()>0
        for r in metrics.itertuples():
            suffix=f"{r.feature_set.lower()}_{'batch' if r.solver=='batch_gd' else r.solver}"
            actual=predictions.ac_power.to_numpy()
            predicted=predictions[f'pred_{suffix}'].to_numpy()
            np.testing.assert_allclose(predicted,np.maximum(predictions[f'raw_pred_{suffix}'],0))
            self.assertAlmostEqual(r.rmse_all_hours,rmse(actual,predicted),places=7)
            self.assertAlmostEqual(r.rmse_daytime,rmse(actual[mask],predicted[mask]),places=7)
        for key in ('a','b'):
            t=pd.read_csv(results/f'table3_set_{key}.csv')
            np.testing.assert_array_equal(t.normal.round(2),t.batch_gd.round(2))
            self.assertLessEqual(np.max(np.abs(t.normal-t.batch_gd)),1e-5)
            with np.load(results/f'set_{key}_normal_weights.npz',allow_pickle=False) as weights:
                self.assertEqual(weights['feature_names'].dtype.kind,'U')

if __name__=='__main__': unittest.main()
