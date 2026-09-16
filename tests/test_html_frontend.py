"""Verify the exported HTML model matches NumPy inference and saved evidence."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from html_server import export_model, export_plots, PLOTS
from prediction import load_weights,predict


class HtmlModelTests(unittest.TestCase):
    def test_all_nine_plots_are_exact_portable_copies(self):
        html=(ROOT/'app'/'web'/'index.html').read_text(encoding='utf-8')
        with tempfile.TemporaryDirectory() as directory:
            names=export_plots(directory)
            self.assertEqual(len(names),9)
            for name in PLOTS:
                self.assertEqual((Path(directory)/name).read_bytes(),(ROOT/'results'/name).read_bytes())
                self.assertIn(f'src="plots/{name}"',html)
                self.assertIn(f'href="plots/{name}"',html)

    def test_export_matches_saved_model_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model-data.js'
            data=export_model(path)
            content=path.read_text(encoding='utf-8')
            parsed=json.loads(content.split('window.SOLAR_MODEL = ',1)[1].strip().removesuffix(';'))
            self.assertEqual(parsed,data)
            self.assertEqual(data['feature_names'],['sw_radiation','temp_2m','cloud_cover','sin_hour','cos_hour'])
            weights=load_weights()
            for hour in range(24):
                for radiation,temp,cloud in ((0,24,15),(400,28,85),(950,35,10),(650,31.2,45)):
                    features=[radiation,temp,cloud,np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24)]
                    raw=data['theta'][0]+sum((v-data['train_means'][i])/data['train_stds'][i]*data['theta'][i+1] for i,v in enumerate(features))
                    expected=predict(hour,radiation,temp,cloud,weights)[3]
                    self.assertAlmostEqual(max(0,raw),expected,places=8)
            self.assertEqual(len(data['hourly_profile']),24)
            self.assertEqual(len(data['metrics']),6)
            self.assertEqual(len(data['test_series']),168)
            self.assertTrue(all(np.isfinite(data['feature_min'])))
            self.assertTrue(all(np.isfinite(data['feature_max'])))

    def test_html_has_all_paired_controls_and_no_remote_assets(self):
        html=(ROOT/'app'/'web'/'index.html').read_text(encoding='utf-8')
        for field in ('hour','sw_radiation','temp_2m','cloud_cover'):
            self.assertIn(f'id="{field}-slider"',html)
            self.assertIn(f'id="{field}-number"',html)
        self.assertNotIn('https://',html)
        self.assertIn('prefers-reduced-motion:reduce',(ROOT/'app'/'web'/'styles.css').read_text())


if __name__=='__main__':
    unittest.main()
