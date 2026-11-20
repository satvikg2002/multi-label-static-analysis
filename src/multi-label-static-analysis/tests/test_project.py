"""Checks for the incremental analysis milestone, without classifier training."""
from pathlib import Path
import json,subprocess,sys,tempfile,unittest
import numpy as np
from multi_label_static_analysis.config import project_paths
from multi_label_static_analysis.data_analysis import binary_anova,feature_dictionary,common_extractors,vectorize_record,assert_no_models,verify_score_reference,stream_metadata
P=project_paths(Path(__file__)); ROOT=P.root
class AnalysisTests(unittest.TestCase):
    def test_project_discovery_and_paths(self):
        self.assertEqual(project_paths(P.notebook.parent).root,ROOT)
        self.assertTrue(P.notebook.is_relative_to(P.project))
        self.assertEqual(P.notebook.name,'ember2024_primary_analysis.ipynb')
        self.assertTrue(P.data.is_dir())
    def test_toml_override(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve(); (root/'pyproject.toml').write_text('[project]\nname="multi-label-static-analysis"\n[tool.multi-label-static-analysis]\ndata-dir="inputs"\n')
            self.assertEqual(project_paths(root).data,root/'inputs')
    def test_anova_matches_scipy(self):
        from scipy.stats import f_oneway
        rng=np.random.default_rng(3); a=rng.normal(size=(120,8)); y=np.arange(120)%2
        actual,p=binary_anova(a,y); expected=f_oneway(a[y==0],a[y==1],axis=0)
        np.testing.assert_allclose(actual,expected.statistic,rtol=1e-12)
        np.testing.assert_allclose(p,expected.pvalue,rtol=1e-12)
        a[:,0]=1; self.assertTrue(np.isnan(binary_anova(a,y)[0][0]))
        self.assertTrue(np.isnan(binary_anova(a,np.zeros(len(y)))[0]).all())
    def test_names_match_vector_layout_and_metadata_excluded(self):
        names=feature_dictionary(); self.assertEqual(len(names),696); self.assertEqual(len(set(names)),696)
        file=next(P.data.glob('*Win64_train.jsonl'))
        with file.open() as f: d=json.loads(f.readline())
        first=vectorize_record(d)
        d.update(label=99,behavior=['invented'],family='invented',first_submission_date=0,detection_ratio='99/99')
        np.testing.assert_array_equal(first,vectorize_record(d))
        self.assertEqual(names[0],'general.size'); self.assertEqual(names[519],'strings.numstrings')
    def test_ingestion_rejects_bad_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'toy_train.jsonl'; p.write_text('{"sha256":"abc","label":1}\nnot-json\n')
            rows,profile,sha,issues,errors=stream_metadata([p])
            self.assertEqual(profile[0]['raw_rows'],2); self.assertTrue(issues); self.assertTrue(errors)
    def test_baseline_preserved_and_models_absent(self):
        self.assertTrue(verify_score_reference(P.historical)['files']); assert_no_models(P.artifacts)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.joblib'; p.touch()
            with self.assertRaises(ValueError): assert_no_models(Path(d))
    def test_builder_analysis_only_and_overwrite_guard(self):
        import nbformat
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'analysis.ipynb'
            args=[sys.executable,'-m','multi_label_static_analysis','--project-dir',str(ROOT),'build','--output',str(out)]
            result=subprocess.run(args,cwd=d,text=True,capture_output=True); self.assertEqual(result.returncode,0,result.stderr)
            n=nbformat.read(out,4); nbformat.validate(n)
            code=[c for c in n.cells if c.cell_type=='code']
            self.assertGreater(len(code), 11)
            self.assertEqual(len(set(c.metadata['stage'] for c in code)), 11)
            self.assertEqual(sum(bool(c.metadata['stage_end']) for c in code), 11)
            for cell in n.cells:
                if cell.cell_type == 'markdown':
                    self.assertLessEqual(len(cell.source.split()), 70)
            for c in code:
                compile(c.source,c.metadata['stage'],'exec')
                self.assertTrue(c.source.startswith('# '))
                self.assertLessEqual(len(c.source.splitlines()), 35)
                self.assertTrue(c.metadata['title'])
                for token in ['joblib.load','joblib.dump','LGBMClassifier(','LogisticRegression(','MLPClassifier(']: self.assertNotIn(token,c.source)
            before=out.read_bytes(); result=subprocess.run(args,cwd=d,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0); self.assertEqual(before,out.read_bytes())
    def test_old_training_flag_disabled(self):
        result=subprocess.run([sys.executable,'-m','multi_label_static_analysis','run','--baseline'],cwd=ROOT,text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)

    def test_acquisition_extraction_and_reuse_without_network(self):
        from contextlib import redirect_stdout
        from dataclasses import replace
        from io import StringIO
        from unittest.mock import patch
        from zipfile import ZipFile
        import hashlib
        from multi_label_static_analysis.data_acquisition import process_archive
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'features'
            data.mkdir()
            archive = root / 'Win64_train.zip'
            payload = b'{"file_type":"Win64"}\n' * 2
            with ZipFile(archive, 'w') as stream:
                stream.writestr('nested/Win64_train.jsonl', payload)
                stream.writestr('ignored.txt', 'Not a feature record')
            item = {'filename': archive.name, 'bytes': archive.stat().st_size,
                    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
            source = {'repo_id': 'local/test', 'revision': 'fixture'}
            complete = {'archives': {}}
            manifest = root / 'completion.json'
            paths = replace(P, data=data)
            with patch('huggingface_hub.hf_hub_download', return_value=str(archive)) as download:
                with redirect_stdout(StringIO()):
                    process_archive(item, paths, source, root, complete, manifest)
                    process_archive(item, paths, source, root, complete, manifest)
                self.assertEqual(download.call_count, 1)
            self.assertEqual((data / 'Win64_train.jsonl').read_bytes(), payload)
            self.assertFalse((data / 'ignored.txt').exists())
            self.assertFalse(archive.exists())
            self.assertEqual(complete['archives'][item['filename']]['extracted'][0]['raw_rows'], 2)

    def test_download_builder_has_short_commented_steps(self):
        from dataclasses import replace
        from unittest.mock import patch
        from contextlib import redirect_stdout
        from io import StringIO
        import runpy
        import nbformat
        with tempfile.TemporaryDirectory() as directory:
            with patch('multi_label_static_analysis.config.project_paths', return_value=replace(P, project=Path(directory))):
                with redirect_stdout(StringIO()):
                    runpy.run_module('multi_label_static_analysis.notebook_builders.build_download_notebook', run_name='__main__')
            notebook = nbformat.read(Path(directory) / 'notebooks/ember2024_win64_download.ipynb', 4)
            cells = [c for c in notebook.cells if c.cell_type == 'code']
            self.assertEqual(len(cells), 6)
            self.assertEqual(sum(c.metadata['stage_end'] for c in cells), 3)
            for cell in cells:
                self.assertLessEqual(len(cell.source.splitlines()), 25)
                self.assertTrue(cell.source.startswith('# '))
                compile(cell.source, cell.metadata['title'], 'exec')
if __name__=='__main__': unittest.main()
