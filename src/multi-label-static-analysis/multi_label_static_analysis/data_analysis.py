"""Small, testable helpers for descriptive analysis; no predictive estimators."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import f as f_distribution

FEATURE_GROUPS = ('general', 'histogram', 'byteentropy', 'strings')
MODEL_SUFFIXES = {'.joblib', '.pkl', '.pickle', '.model', '.pt', '.pth', '.onnx', '.h5', '.ckpt', '.safetensors'}


def stream_metadata(paths):
    """Read each JSONL row once, retain metadata only, and audit group schemas."""
    rows=[]; profile=[]; source_hash=hashlib.sha256(); missing=Counter(); malformed=[]
    for path in paths:
        official='challenge' if 'challenge' in path.name else 'test' if '_test' in path.name else 'train'
        n=0; type_counts=Counter()
        with path.open() as stream:
            for line_no,line in enumerate(stream):
                source_hash.update(line.encode('utf-8')); n+=1
                try:
                    d=json.loads(line)
                    if not isinstance(d,dict) or not isinstance(d.get('sha256'),str): raise ValueError('missing identity')
                    tags=tuple(sorted({str(t).lower().strip() for t in (d.get('behavior') or [])}))
                    general=d.get('general') or {}
                    rows.append({'sha256':d['sha256'],'tlsh':d.get('tlsh'),'timestamp':d.get('first_submission_date'),
                        'official':official,'file_type':d.get('file_type'),'malicious':d.get('label'),
                        'family':d.get('family'),'tags':tags,'source':path.name,'line':line_no,
                        'file_size':general.get('size'),'byte_entropy':general.get('entropy'),
                        'family_confidence':d.get('family_confidence')})
                    type_counts.update([str(d.get('file_type'))])
                    for key in FEATURE_GROUPS:
                        if key not in d or d[key] is None: missing[key]+=1
                    if len(d.get('histogram',[]))!=256: missing['histogram_wrong_length']+=1
                    if len(d.get('byteentropy',[]))!=256: missing['byteentropy_wrong_length']+=1
                except (ValueError,TypeError,KeyError) as error:
                    malformed.append({'source':path.name,'line':line_no,'error':str(error)})
        profile.append({'file':path.name,'official':official,'bytes':path.stat().st_size,
                        'mtime_ns':path.stat().st_mtime_ns,'raw_rows':n,'file_types':dict(type_counts)})
        if len(profile)%32==0: print('Ingested',len(profile),'of',len(paths),'files;',len(rows),'metadata rows',flush=True)
    return rows,profile,source_hash.hexdigest(),dict(missing),malformed


def common_extractors():
    from . import thrember
    return [thrember.GeneralFileInfo(),thrember.ByteHistogram(),thrember.ByteEntropyHistogram(),thrember.StringExtractor()]


def vectorize_record(record, extractors=None):
    """Allowlist the four static groups; target/identity metadata cannot enter X."""
    extractors=extractors or common_extractors()
    with np.errstate(divide='ignore',invalid='ignore'):
        vector=np.concatenate([ex.process_raw_features(record[ex.name]) for ex in extractors]).astype(np.float32)
    return np.where(np.isfinite(vector),vector,np.nan)


def feature_dictionary(extractors=None):
    extractors=extractors or common_extractors()
    names=['general.size','general.entropy','general.is_pe']+[f'general.start_byte_{i}' for i in range(4)]
    names += [f'histogram.byte_{i:03d}' for i in range(256)]
    names += [f'byteentropy.entropy_{i:02d}.bytebin_{j:02d}' for i in range(16) for j in range(16)]
    names += ['strings.numstrings','strings.avlength','strings.printables']
    names += [f'strings.printable_{i+32:03d}' for i in range(96)]
    names += ['strings.entropy']
    string_extractor=extractors[-1]
    names += ['strings.count.'+name for name,index in sorted(string_extractor.regex_idxs.items(),key=lambda p:p[1])]
    assert len(names)==sum(ex.dim for ex in extractors)
    return names


def binary_anova(values, labels):
    """Exact binary ANOVA for one bounded column block; no classifier is fitted."""
    a=np.asarray(values,dtype=np.float64); y=np.asarray(labels)
    if a.ndim!=2 or y.ndim!=1 or len(a)!=len(y): raise ValueError('Incompatible feature/target shapes')
    if not np.isfinite(a).all(): raise ValueError('Impute fitting values before ANOVA')
    if not set(np.unique(y)) <= {0,1}: raise ValueError('Expected binary observed labels')
    n0=int((y==0).sum()); n1=int((y==1).sum())
    if not n0 or not n1 or len(y)<=2: return np.full(a.shape[1],np.nan),np.full(a.shape[1],np.nan)
    a0=a[y==0]; a1=a[y==1]
    mean0=a0.mean(axis=0); mean1=a1.mean(axis=0)
    # Centered residuals avoid cancellation when large size/count values are close.
    within=np.sum((a0-mean0)**2,axis=0)+np.sum((a1-mean1)**2,axis=0)
    between=(mean0-mean1)**2*n0*n1/(n0+n1)
    constant=np.ptp(a,axis=0)==0
    with np.errstate(divide='ignore',invalid='ignore'):
        scores=between/(within/(n0+n1-2))
    scores[constant]=np.nan
    p=f_distribution.sf(scores,1,n0+n1-2)
    return scores,p


def verify_score_reference(directory):
    manifest=json.loads((directory/'baseline_score_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Historical score changed: {name}')
    return manifest


def assert_no_models(directory):
    found=[str(p) for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in MODEL_SUFFIXES]
    if found: raise ValueError('Trained model files remain: '+', '.join(found))
