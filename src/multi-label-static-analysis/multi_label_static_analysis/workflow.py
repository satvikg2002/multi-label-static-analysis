"""Execute the notebook and check its saved research outputs."""
import hashlib
import json
import os
from pathlib import Path
import sys

from .config import ProjectPaths


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def run_notebook(paths: ProjectPaths, *, interactive: bool = False,
                 preprocessing_only: bool = False, timeout: int = 7200,
                 require_data: bool = True) -> None:
    import nbformat
    from nbclient import NotebookClient

    if not paths.notebook.is_file():
        raise FileNotFoundError(f'Notebook not found: {paths.notebook}')
    if require_data and not any(paths.data.glob('*.jsonl')):
        raise FileNotFoundError(f'No extracted EMBER2024 JSONL files in {paths.data}')
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / 'figures').mkdir(parents=True, exist_ok=True)
    kernel_root = paths.artifacts / 'jupyter'
    kernel_dir = kernel_root / 'kernels' / 'msai603'
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'MSAI 603 (uv project)', 'language': 'python',
    }, indent=2))
    # Keep runtime files in this project. Use the interpreter selected by uv,
    # rather than assuming a particular .venv path or touching global kernels.
    environment = {
        'JUPYTER_PATH': str(kernel_root) + (os.pathsep + os.environ['JUPYTER_PATH'] if os.environ.get('JUPYTER_PATH') else ''),
        'IPYTHONDIR': str(paths.artifacts / 'ipython'),
        'MPLCONFIGDIR': str(paths.artifacts / 'matplotlib'),
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    notebook = nbformat.read(paths.notebook, as_version=4)
    class StreamingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg.get('msg_type') == 'stream':
                print(msg['content'].get('text', ''), end='', flush=True)
            return super().process_message(msg, cell, cell_index)
    client = StreamingClient(notebook, timeout=timeout, kernel_name='msai603',
                             resources={'metadata': {'path': str(paths.root)}})
    try:
        with client.setup_kernel():
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                stage = cell.metadata.get('stage', f'cell_{index}')
                if preprocessing_only and stage.startswith('5_'):
                    break
                step = cell.metadata.get('step', 1)
                if step == 1:
                    print(f'RUNNING {stage}', flush=True)
                if 'title' in cell.metadata:
                    print(f'  {step}. {cell.metadata["title"]}', flush=True)
                client.execute_cell(cell, index)
                nbformat.write(notebook, paths.notebook)
                if cell.metadata.get('stage_end', True):
                    print(f'STAGE COMPLETE {stage}', flush=True)
                if interactive:
                    input('Press Enter for next stage: ')
    finally:
        nbformat.write(notebook, paths.notebook)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    print('PREPROCESSING VERIFIED' if preprocessing_only else 'NOTEBOOK COMPLETE', flush=True)


def baseline_scores(paths: ProjectPaths):
    """Read the protected historical scores; never load fitted classifiers."""
    import pandas as pd
    from .data_analysis import verify_score_reference
    verify_score_reference(paths.historical)
    return pd.read_csv(paths.historical/'ember2024_baseline_comparison.csv')


def verify_outputs(paths: ProjectPaths) -> None:
    import json
    import nbformat
    import numpy as np
    from .data_analysis import assert_no_models, verify_score_reference
    notebook=nbformat.read(paths.notebook,as_version=4); nbformat.validate(notebook)
    expected=['1_scope_inventory','2_ingestion','3_identity_time_audit','4_label_analysis','5_vectorization',
              '6_feature_statistics','7_normalization_redundancy','8_feature_selection','9_pca_visualization',
              '10_distribution_drift','11_findings_reference']
    cells=[c for c in notebook.cells if c.cell_type=='code']
    # A section now consists of several small cells, each with a named task.
    stages = list(dict.fromkeys(c.metadata.get('stage') for c in cells))
    if stages != expected:
        raise ValueError('Unexpected primary analysis stages')
    for stage in expected:
        steps = [c for c in cells if c.metadata['stage'] == stage]
        if [c.metadata.get('step') for c in steps] != list(range(1, len(steps) + 1)):
            raise ValueError(f'Incomplete step sequence: {stage}')
        if any(c.metadata.get('stage_end') for c in steps[:-1]) or not steps[-1].metadata.get('stage_end'):
            raise ValueError(f'Invalid section boundary: {stage}')
    for cell in cells:
        if cell.execution_count is None or any(o.output_type=='error' for o in cell.outputs):
            raise ValueError(f'Incomplete stage: {cell.metadata["stage"]}')
        if cell.metadata.get('stage_end') and not any(o.output_type=='stream' and 'PASS:' in o.text for o in cell.outputs):
            raise ValueError(f'Missing stage check: {cell.metadata["stage"]}')
        if any(token in cell.source for token in ['joblib.load','joblib.dump','LogisticRegression(', 'LGBMClassifier(', 'MLPClassifier(', 'SVC(']):
            raise ValueError('Predictive-model code found in active analysis notebook')
    assert_no_models(paths.artifacts); verify_score_reference(paths.historical)
    summary=json.loads((paths.reports/'analysis_summary.json').read_text())
    if summary['classifiers_trained']!=0 or summary['serialized_models']!=0: raise ValueError('Analysis-only contract violated')
    X=np.load(paths.artifacts/'static_features.npy',mmap_mode='r')
    if X.shape!=(summary['retained_metadata_rows'],summary['features']): raise ValueError('Vector cache shape mismatch')
    selection=json.loads((paths.reports/'candidate_feature_subsets.json').read_text())
    for k in [64,128,256]:
        values=selection['candidates'][f'top_{k}']
        if len(values)!=k or len(set(values))!=k: raise ValueError('Invalid proposed feature subset')
    if len(list((paths.reports/'figures').glob('*.png')))<9: raise ValueError('Missing analysis visualizations')
    print('PASS: all 11 primary-analysis stages executed with saved outputs and no classifier code.')
    print('PASS: vector cache, proposed feature subsets and visualizations verified.')
    print('PASS: previous fitted models absent; historical baseline score checksums unchanged.')
