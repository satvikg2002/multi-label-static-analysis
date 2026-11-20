"""Generate the pinned Win64 feature-data acquisition notebook."""
from pathlib import Path
import nbformat as nbf
from multi_label_static_analysis.config import project_paths
p=project_paths(); out=p.project/'notebooks/ember2024_win64_download.ipynb'
nb=nbf.v4.new_notebook(); nb.metadata.kernelspec={'name':'python3','display_name':'Python (uv project)','language':'python'}
cells=[nbf.v4.new_markdown_cell('''# Add EMBER2024 Win64 training and test features

Download **feature JSONL archives**, not executable binaries, from the official [EMBER2024 dataset](https://huggingface.co/datasets/joyce8/EMBER2024). The checked-in `configs/ember2024/win64_sources.json` pins the dataset revision, archive lengths, and SHA256 hashes. Downloaded archives are hashed before extraction; only JSONL members are extracted into the existing data directory. Per-file hashes, row counts, sizes, and timestamps are recorded in a completion manifest. Archives are removed after verified extraction to avoid retaining duplicate multi-gigabyte copies.

Both official train and test splits are added. The research notebook will split training chronologically into fitting, validation, and calibration, and keep the official test split held out. Duplicate identities and shared-vector leakage are audited in the research notebook.''')]
def code(source,stage):
 c=nbf.v4.new_code_cell(source); c.metadata['stage']=stage; cells.append(c)
code('''from pathlib import Path
import os,json,hashlib,shutil,zipfile,time
from multi_label_static_analysis.config import project_paths
P=project_paths(); P.data.mkdir(parents=True,exist_ok=True)
SOURCE=json.loads((P.project/'configs/ember2024/win64_sources.json').read_text())
DOWNLOAD=P.artifacts/'downloads'; DOWNLOAD.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(DOWNLOAD/'hf_home')
os.environ['HF_XET_CACHE']=str(DOWNLOAD/'xet')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
disable_progress_bars()
print('Pinned dataset revision:',SOURCE['revision'])
print('Archives GB:',sum(f['bytes'] for f in SOURCE['files'])/1e9)
assert shutil.disk_usage(P.root).free>35*10**9, 'Need at least 35 GB free for archives, features, and expanded model inputs.'
print('PASS: pinned Win64 sources and disk capacity checked.')
''','1_download_setup')
code('''def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
completed_path=P.artifacts/'win64_download_manifest.json'
completed=json.loads(completed_path.read_text()) if completed_path.exists() else {'revision':SOURCE['revision'],'archives':{}}
assert completed['revision']==SOURCE['revision']
for item in sorted(SOURCE['files'],key=lambda f:'_train' not in f['filename']):
    previous=completed['archives'].get(item['filename'])
    if previous and previous['sha256']==item['sha256']:
        valid=True
        for entry in previous['extracted']:
            dest=P.data/entry['file']
            if not dest.exists() or dest.stat().st_size!=entry['bytes']:
                valid=False; break
            if dest.stat().st_mtime_ns!=entry['mtime_ns'] and file_sha(dest)!=entry['sha256']:
                valid=False; break
        if valid:
            print('Reusing verified extracted archive:',item['filename'],flush=True); continue
    print('Downloading',item['filename'],f"({item['bytes']/1e9:.2f} GB)",flush=True)
    archive=Path(hf_hub_download(repo_id=SOURCE['repo_id'],repo_type='dataset',revision=SOURCE['revision'],
                    filename=item['filename'],local_dir=str(DOWNLOAD)))
    assert archive.stat().st_size==item['bytes']
    assert file_sha(archive)==item['sha256'], 'Archive SHA256 mismatch'
    print('Archive checksum passed; extracting',item['filename'],flush=True)
    extracted=[]
    with zipfile.ZipFile(archive) as z:
        members=[info for info in z.infolist() if not info.is_dir() and info.filename.endswith('.jsonl')]
        assert members, 'No feature JSONL files in archive'
        assert len({Path(info.filename).name for info in members})==len(members)
        for info in members:
            # Extract only the base filename; never honor archive directory traversal.
            dest=P.data/Path(info.filename).name
            assert 'Win64' in dest.name
            partial=dest.with_suffix('.jsonl.partial'); h=hashlib.sha256(); lines=0; last=b''
            with z.open(info) as source, partial.open('wb') as target:
                for chunk in iter(lambda:source.read(8*1024*1024),b''):
                    target.write(chunk); h.update(chunk); lines+=chunk.count(b'\\n'); last=chunk[-1:]
            assert partial.stat().st_size==info.file_size
            if last and last!=b'\\n': lines+=1
            partial.replace(dest)
            extracted.append({'file':dest.name,'bytes':info.file_size,'sha256':h.hexdigest(),
                              'raw_rows':lines,'mtime_ns':dest.stat().st_mtime_ns})
    completed['archives'][item['filename']]={'sha256':item['sha256'],'bytes':item['bytes'],'extracted':extracted}
    completed_path.write_text(json.dumps(completed,indent=2))
    archive.unlink()
    print('Completed',item['filename'],'files:',len(extracted),'raw rows:',sum(f['raw_rows'] for f in extracted),flush=True)
print('PASS: both archives verified and JSONL feature files extracted; completion manifest saved.')
''','2_download_extract')
code('''for name,minimum in [('Win64_train.zip',520000),('Win64_test.zip',120000)]:
    info=completed['archives'][name]
    rows=sum(entry['raw_rows'] for entry in info['extracted'])
    assert rows>=minimum
    for entry in info['extracted']:
        path=P.data/entry['file']; assert path.stat().st_size==entry['bytes']
        with path.open() as stream: example=json.loads(stream.readline())
        assert example['file_type']=='Win64'
    print(name,'verified JSONL files:',len(info['extracted']),'raw rows:',rows)
print('PASS: Win64 train/test feature schema and expected row-count coverage checked; deduplication follows in research audit.')
''','3_download_verify')
from multi_label_static_analysis.notebook_readability import readable_notebook
nb.cells=cells
nb=readable_notebook(nb,kind='download'); out.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,out); print(out)
