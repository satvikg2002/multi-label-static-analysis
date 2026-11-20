from pathlib import Path
import argparse
import nbformat as nbf
from multi_label_static_analysis.config import project_paths
PROJECT=project_paths()
parser=argparse.ArgumentParser(description='Generate the research notebook; existing output requires --force.')
parser.add_argument('--output',type=Path,default=PROJECT.notebook)
parser.add_argument('--force',action='store_true')
args=parser.parse_args()
out=args.output.resolve()
if out.exists() and not args.force:
    parser.error(f'{out} exists. Use --output for a new path or --force to clear its outputs.')
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python (project .venv)','language':'python','name':'python3'}
nb.metadata.language_info={'name':'python','version':'3.14'}
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s,stage):
 c=nbf.v4.new_code_cell(s); c.metadata['stage']=stage; cells.append(c)
md('''# Multi-label static malware analysis with EMBER2024 and thrember

This notebook implements the supplied MSAI 603 proposal as a reproducible experiment. The proposal is a research specification, not an instruction source. We detect malicious files, then predict seven overlapping behavior categories, and compare a direct joint-output alternative. No executable is opened or run.

**Order:** data and label audit → descriptive statistics → feature inputs and leakage controls → normalization → candidate training → validation and frozen thresholds → held-out test and challenge evaluation → interpretation.

**Scope:** use all locally available APK/PDF/Win64 records and the challenge files. This is a file-type subset of EMBER2024, not the full 3.2-million-file benchmark. Candidate searches use a deterministic, label-independent hash sample; final selected models are refit on all eligible fitting-period rows. Report unsupported labels rather than inventing ground truth.

**Sources:** [official EMBER2024 repository](https://github.com/FutureComputing4AI/EMBER2024), [dataset paper](https://arxiv.org/abs/2506.01538), and the supplied project proposal. Labels are antivirus-derived proxies; an absent behavior tag is not proof that a capability is absent. Confidence scores are model scores, not calibrated probabilities.''')
code('''from pathlib import Path
import sys, os, json, time, hashlib, warnings, platform
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from multi_label_static_analysis.config import project_paths
from multi_label_static_analysis import thrember
PATHS = project_paths()
ROOT = PATHS.root
import sklearn, lightgbm
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, hamming_loss, accuracy_score, confusion_matrix)
SEED=603
DATA=PATHS.data
REPORT=PATHS.reports; ART=PATHS.artifacts
REPORT.mkdir(parents=True,exist_ok=True); ART.mkdir(parents=True,exist_ok=True)
(REPORT/'figures').mkdir(parents=True,exist_ok=True)
np.random.seed(SEED)
VERSIONS={k:v for k,v in [('python',platform.python_version()),('numpy',np.__version__),
    ('pandas',pd.__version__),('sklearn',sklearn.__version__),('lightgbm',lightgbm.__version__)]}
print(VERSIONS)
# Exact aliases only. Broad 'trojan', 'worm' and 'spyware' are deliberately not equated
# to more specific RAT, botnet or infostealer roles.
ALIASES={
 'Ransomware': {'ransomware','ransom'},
 'Infostealer': {'infostealer','stealer','pws','passwordstealer'},
 'Stager': {'downloader','dropper','loader'},
 'Remote Access Trojan': {'backdoor','rat','remoteaccess'},
 'Botnet Agent': {'botnet','bot'},
 'Banking Trojan': {'banker','banking','bankingtrojan'},
 'Cryptominer': {'miner','coinminer','cryptominer','cryptocurrencyminer'},
}
LABELS=list(ALIASES)
paths=sorted(DATA.glob('*.jsonl'))
assert paths, 'Download EMBER2024 extracted feature JSONL files first.'
manifest=[{'file':p.name,'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths]
rows=[]; behavior_counts=Counter(); fingerprint=hashlib.sha256()
for p in paths:
    official='challenge' if 'challenge' in p.name else 'test' if '_test' in p.name else 'train'
    with p.open() as f:
        for line_no,line in enumerate(f):
            d=json.loads(line)
            tags={str(t).lower().strip() for t in (d.get('behavior') or [])}
            if official=='train': behavior_counts.update(tags)
            rows.append({'sha256':d['sha256'],'tlsh':d.get('tlsh'),'timestamp':d.get('first_submission_date'),
                'official':official,'file_type':d.get('file_type'),'malicious':d.get('label'),
                'family':d.get('family'),'tags':tuple(sorted(tags)),'source':p.name,'line':line_no})
            fingerprint.update((d['sha256']+'|'+str(d.get('first_submission_date'))+'|'+official+'\\n').encode())
meta=pd.DataFrame(rows); del rows
meta['date']=pd.to_datetime(meta.timestamp,unit='s',utc=True,errors='coerce')
assert meta.date.notna().all(), 'Missing dates require an explicit exclusion policy.'
assert set(meta.malicious.unique()) <= {0,1}, 'Unknown binary labels need exclusion.'
# Split by the true record timestamp, never filename order or row order.
train_dates=meta.loc[meta.official.eq('train'),'date']
train_end=train_dates.max().normalize()+pd.Timedelta(days=1)
val_start=train_end-pd.Timedelta(weeks=12)
cal_start=train_end-pd.Timedelta(weeks=6)
meta['split']=meta.official
meta.loc[meta.official.eq('train') & meta.date.ge(val_start),'split']='validation'
meta.loc[meta.official.eq('train') & meta.date.ge(cal_start),'split']='calibration'
meta.loc[meta.split.eq('train'),'split']='fit'
# Earliest timestamp wins. Never move a future observation into an earlier split.
meta=meta.sort_values(['date','sha256','source','line']).reset_index(drop=True)
duplicate_count=int(meta.sha256.duplicated().sum())
meta=meta.drop_duplicates('sha256',keep='first').copy()
# Exact valid TLSH equality is a conservative similarity grouping. Keep each group
# in its earliest chronological partition and purge subsequent members.
order={'fit':0,'validation':1,'calibration':2,'test':3,'challenge':4}
meta['priority']=meta.split.map(order)
valid_tlsh=meta.tlsh.notna() & ~meta.tlsh.isin(['','TNULL','NULL','-'])
group_first=meta.loc[valid_tlsh].groupby('tlsh').priority.min()
leak=valid_tlsh & meta.priority.gt(meta.tlsh.map(group_first))
purged=int(leak.sum()); meta=meta.loc[~leak].reset_index(drop=True)
meta['row_id']=np.arange(len(meta))
assert meta.sha256.is_unique
for s in ['fit','validation','calibration','test']:
    assert (meta.split==s).any(), f'Empty {s} split'
for a,b in [('fit','validation'),('validation','calibration'),('calibration','test')]:
    assert meta.loc[meta.split.eq(a),'date'].max() < meta.loc[meta.split.eq(b),'date'].min()
    aa=set(meta.loc[meta.split.eq(a)&valid_tlsh.reindex(meta.index,fill_value=False),'tlsh'].dropna()) if False else None
# Recompute grouping audit after index reset.
valid=meta.tlsh.notna() & ~meta.tlsh.isin(['','TNULL','NULL','-'])
assert meta.loc[valid].groupby('tlsh').split.nunique().max()==1
print(f'{len(paths)} files; {len(meta):,} retained records; {duplicate_count:,} SHA duplicates removed; {purged:,} cross-split exact-TLSH rows purged')
display(meta.groupby(['split','file_type']).agg(n=('sha256','size'),start=('date','min'),end=('date','max')))
display(pd.Series(behavior_counts,name='training tag count').sort_values(ascending=False).head(40))
(ART/'data_manifest.json').write_text(json.dumps({'files':manifest,'ordered_metadata_sha256':fingerprint.hexdigest(),
 'versions':VERSIONS,'aliases':{k:sorted(v) for k,v in ALIASES.items()},'sha_duplicates':duplicate_count,'tlsh_purged':purged,
 'thrember_features_sha256':hashlib.sha256(Path(thrember.__file__).with_name('features.py').read_bytes()).hexdigest()},indent=2))
import importlib.metadata as package_metadata
packages=['numpy','pandas','scipy','scikit-learn','lightgbm','matplotlib','nbformat','nbclient','ipykernel','pypdf','pefile','signify','polars','huggingface-hub','joblib']
(REPORT/'environment.json').write_text(json.dumps({p:package_metadata.version(p) for p in packages},indent=2))
print('PASS: metadata schema, unique identities, chronological splits, exact TLSH separation.')
''','1_data_audit')
md('''## 2. Statistical analysis and target feasibility

Describe fitting data before model design. Show class balance, behavior prevalence, multi-label cardinality and co-occurrence. Seven targets are mapped only from explicit behavior aliases; family names, detection ratios, Capa outputs and confidence annotations never become predictors. Fit/validation periods determine which labels are estimable. Empty behavior lists on malicious files mean **unannotated**, not seven verified negatives: exclude them from behavior training and scored category evaluation. Malicious files tagged only with other behaviors remain observed negatives for these seven proxy targets, an imperfect closed-world assumption.

Missing and rare categories remain in support reports. They are excluded from model-selection macro averages and marked unavailable when there are too few positive and negative fitting examples. Categories unsupported in a particular evaluation split have undefined AP and recall, not zero AP.''')
code('''Y=np.array([[int(bool(set(tags)&ALIASES[label])) for label in LABELS] for tags in meta.tags],dtype=np.int8)
y=meta.malicious.to_numpy(dtype=np.int8)
annotated=np.array([bool(t) for t in meta.tags])
# Benign rows are explicit all-zero behavior targets; only annotated malware is
# scored for behaviors. Gate metrics always use every eligible labeled record.
behavior_observed=(y==0)|annotated
fit_mask=meta.split.eq('fit').to_numpy()
fit_behavior=fit_mask & behavior_observed
support=pd.DataFrame({'category':LABELS,'fit_positive':Y[fit_behavior].sum(axis=0),
    'fit_negative':len(Y[fit_behavior])-Y[fit_behavior].sum(axis=0),
    'annotated_malware_fit_positive':Y[fit_mask & (y==1) & annotated].sum(axis=0)})
active=(support.fit_positive.to_numpy()>=20)&(support.fit_negative.to_numpy()>=20)
assert active.any(), 'No proposed categories are estimable in this file-type subset.'
support['estimable']=active
display(support); support.to_csv(REPORT/'target_support.csv',index=False)
train_stats=meta[fit_mask].groupby('file_type').agg(n=('sha256','size'),malware_rate=('malicious','mean'))
display(train_stats)
print('Missing behavior annotations among fitting malware:',int((fit_mask & (y==1) & ~annotated).sum()))
print('Multi-label cardinality among annotated fitting malware:',Y[fit_mask & (y==1) & annotated].sum(axis=1).mean())
fig,axes=plt.subplots(1,3,figsize=(16,4))
train_stats.n.plot.bar(ax=axes[0],title='Fitting records by format')
support.set_index('category').fit_positive.plot.barh(ax=axes[1],title='Positive proxy-label support (fit)')
co=Y[fit_mask].astype(np.int64).T@Y[fit_mask].astype(np.int64)
im=axes[2].imshow(co,cmap='Blues'); axes[2].set_xticks(range(7),LABELS,rotation=90); axes[2].set_yticks(range(7),LABELS)
axes[2].set_title('Fitting-label co-occurrence'); fig.colorbar(im,ax=axes[2]); plt.tight_layout()
fig.savefig(REPORT/'figures/01_data_audit.png',dpi=150,bbox_inches='tight'); plt.show()
print('PASS: seven-column binary target matrix; support eligibility frozen using fitting data only.')
''','2_statistics')
md('''## 3. Convert raw static features into numerical inputs

Use thrember's `GeneralFileInfo`, `ByteHistogram`, `ByteEntropyHistogram` and `StringExtractor`, the shared cross-format feature groups. PE-specific features are available for Win64 but absent from APK/PDF. This run retains the same 696 shared inputs across APK, PDF and Win64; challenge Win32, .NET and ELF remain outside the fitting formats. No target metadata is passed to these extractors. Histogram frequencies already sum to one; they still need train-fitted scaling for linear models.

Vectorization is streamed once and cached with its source/extractor fingerprint. Non-finite numbers become missing values, handled downstream by a fitting-only median imputer. Invalid records are logged and fail the stage rather than silently disappearing. Exact vector duplicates crossing chronological partitions are additionally purged. Exact TLSH/vector equality does not guarantee separation of all near-duplicate binaries: fuzzy-distance clustering and family-disjoint evaluation are limitations.''')
code('''extractors=[thrember.GeneralFileInfo(),thrember.ByteHistogram(),thrember.ByteEntropyHistogram(),thrember.StringExtractor()]
groups={}; offset=0
for ex in extractors:
    groups[ex.name]=list(range(offset,offset+ex.dim)); offset+=ex.dim
D=offset
key=hashlib.sha256((fingerprint.hexdigest()+str(meta[['source','line']].values.tolist())+
    hashlib.sha256(Path(thrember.__file__).with_name('features.py').read_bytes()).hexdigest()).encode()).hexdigest()
cache=ART/'static_features.npy'; keyfile=ART/'feature_cache_key.txt'
if cache.exists() and keyfile.exists() and keyfile.read_text()==key:
    X=np.load(cache,mmap_mode='r'); print('Loaded verified vector cache',X.shape)
else:
    X=np.lib.format.open_memmap(cache,mode='w+',dtype='float32',shape=(len(meta),D))
    lookup={(r.source,int(r.line)):int(r.row_id) for r in meta.itertuples()}
    n=0; invalid=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        for p in paths:
            with p.open() as f:
                for line_no,line in enumerate(f):
                    idx=lookup.get((p.name,line_no))
                    if idx is None: continue
                    d=json.loads(line)
                    # Each extractor receives only its own allowlisted feature group.
                    v=np.concatenate([ex.process_raw_features(d[ex.name]) for ex in extractors])
                    assert v.shape==(D,)
                    invalid+=int((~np.isfinite(v)).sum())
                    X[idx]=np.where(np.isfinite(v),v,np.nan); n+=1
    X.flush(); assert n==len(meta); keyfile.write_text(key)
    print('Vectorized',n,'rows;',invalid,'non-finite values converted to missing')
assert X.shape==(len(meta),D)
# Conservatively prevent identical vectors crossing partitions, without using labels.
vector_groups={}; keep=np.ones(len(meta),dtype=bool)
for i in np.argsort(meta.priority.to_numpy(),kind='stable'):
    h=hashlib.blake2b(np.nan_to_num(X[i],nan=-1).tobytes(),digest_size=16).digest()
    previous=vector_groups.setdefault(h,meta.split.iloc[i])
    if previous!=meta.split.iloc[i]: keep[i]=False
print('Cross-split exact-vector records purged:',int((~keep).sum()))
meta['eligible']=keep
# Separate logical indexing from stored feature-row indexing; retain original IDs.
idx={s:np.flatnonzero(meta.split.eq(s).to_numpy()&keep) for s in order}
assert all(len(idx[s]) for s in order)
for s in idx: print(s,len(idx[s]))
print('Feature groups:',{k:len(v) for k,v in groups.items()},'total:',D)
print('Missing feature cells:',sum(int(np.isnan(X[start:start+4096]).sum()) for start in range(0,len(X),4096)))
assert D==sum(ex.dim for ex in extractors)
(ART/'feature_groups.json').write_text(json.dumps(groups,indent=2))
meta.loc[keep].groupby(['split','file_type']).agg(n=('sha256','size'),start=('date','min'),end=('date','max')).reset_index().to_csv(REPORT/'split_filetype_counts.csv',index=False)
print('PASS: aligned static feature matrix; metadata excluded; cross-split vector duplicates purged.')
''','3_vectorization')
md('''## 4. Normalization and reproducible search design

Imputation, variance filtering, optional ANOVA feature selection, and scaling live inside each model pipeline and are fit only on its training rows. Compare all shared feature groups with a compact general+strings subset and a train-selected subset. Trees use raw imputed values; linear/SVM/neural models use standard scaling. Class weighting handles imbalance for supported estimators. A most-frequent dummy provides a sanity baseline.

Select up to 30,000 fitting rows using SHA256 order, independent of their labels. All candidates use this same search sample, with a 10,000-row fitting subsample for the nonlinear kernel SVM. This resource cap is disclosed; the final winner is trained on all eligible fitting records. Time-forward validation uses the same label-independent hash sample of at most 30,000 validation records for all candidate models. Final threshold calibration and held-out scoring use their full eligible splits. The linear SVM uses dual optimization with a 1,000-iteration cap and recorded convergence status. Time-forward validation is used instead of random cross-validation. A separate subsequent calibration period chooses decision thresholds after configuration selection; the official test and challenge sets remain unused until evaluation.''')
code('''from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
import joblib
# Existing metadata are already hash-independent chronological records.
hash_order=sorted(idx['fit'],key=lambda i:hashlib.sha256((str(SEED)+meta.sha256.iloc[i]).encode()).digest())
search_idx=np.array(hash_order[:30000]); svm_idx=np.array(hash_order[:10000])
compact=groups['general']+groups['strings']
CONFIGS=[
 dict(name='dummy',kind='dummy',features='all'),
 dict(name='lr_C0.1_all',kind='lr',C=.1,features='all'),
 dict(name='lr_C1_compact',kind='lr',C=1.,features='compact'),
 dict(name='lr_C1_k256',kind='lr',C=1.,features='k256'),
 dict(name='linear_svm_C0.1',kind='svm',C=.1,features='all'),
 dict(name='rbf_svm_C1',kind='rbf',C=1.,features='all'),
 dict(name='forest_depth12',kind='forest',depth=12,features='all'),
 dict(name='forest_depth24',kind='forest',depth=24,features='compact'),
 dict(name='lgbm_leaves15',kind='lgbm',leaves=15,features='all'),
 dict(name='lgbm_leaves31',kind='lgbm',leaves=31,features='all'),
 dict(name='lgbm_compact',kind='lgbm',leaves=15,features='compact'),
 dict(name='mlp_64',kind='mlp',features='all'),
]
def column_subset(a,columns): return a[:,columns]
def pipeline(c):
    steps=[]
    if c['features']=='compact': steps.append(('groups',FunctionTransformer(column_subset,kw_args={'columns':compact})))
    steps += [('impute',SimpleImputer(strategy='median',keep_empty_features=True,copy=False)),('variance',VarianceThreshold())]
    if c['features']=='k256': steps.append(('select',SelectKBest(f_classif,k=256)))
    if c['kind'] in ['lr','svm','rbf','mlp']: steps.append(('scale',StandardScaler(copy=False)))
    k=c['kind']
    if k=='dummy': m=DummyClassifier(strategy='prior')
    elif k=='lr': m=LogisticRegression(C=c['C'],max_iter=350,class_weight='balanced',solver='lbfgs',random_state=SEED)
    elif k=='svm': m=LinearSVC(C=c['C'],class_weight='balanced',dual=True,max_iter=1000,tol=1e-3,random_state=SEED)
    elif k=='rbf': m=SVC(C=c['C'],kernel='rbf',class_weight='balanced',cache_size=512,random_state=SEED)
    elif k=='forest': m=RandomForestClassifier(n_estimators=100,max_depth=c['depth'],min_samples_leaf=5,class_weight='balanced_subsample',n_jobs=4,random_state=SEED)
    elif k=='lgbm': m=LGBMClassifier(n_estimators=180,num_leaves=c['leaves'],learning_rate=.06,min_child_samples=30,
        colsample_bytree=.85,reg_lambda=1.,class_weight='balanced',n_jobs=4,verbosity=-1,random_state=SEED)
    else: m=MLPClassifier(hidden_layer_sizes=(64,),max_iter=45,batch_size=256,alpha=.01,
        early_stopping=False,random_state=SEED)
    return Pipeline(steps+[('model',m)])
normalizer=pipeline(CONFIGS[1])[:-1]
Z=normalizer.fit_transform(X[search_idx],y[search_idx])
assert np.isfinite(Z).all()
assert np.max(np.abs(Z.mean(axis=0)))<.03
print('Normalized sample shape',Z.shape,'largest absolute column mean',np.abs(Z.mean(axis=0)).max())
print('Search rows:',len(search_idx),'RBF SVM rows:',len(svm_idx),'final fitting rows:',len(idx['fit']))
print('PASS: normalization finite and fitted on fitting sample only; no future rows used.')
''','4_normalization')
md('''## 5. Train candidate models

Compare logistic regression, linear and RBF SVMs, random forests, LightGBM and a shallow neural network. The logistic regression gate plus one logistic regression per behavior is the proposed baseline. Binary relevance preserves simultaneous labels rather than forcing one class. Every learned transformation belongs to its estimator pipeline.

Gate models train on all sampled binary-labeled rows. Category models train on sampled **annotated malicious** rows, using other observed behaviors as proxy negatives. A category missing one class in the search sample gets a constant head with an explicit record in the results. A direct shared-hidden-layer MLP predicts maliciousness and the seven category outputs jointly; its training uses benign and annotated-malicious rows because sklearn's MLP has no masked-label loss. Its reduced training coverage is a comparison limitation. Internal random early stopping is disabled to preserve chronological validation.''')
code('''from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
fit_log=[]; gate_models={}; category_models={}; failures=[]
active_ids=np.flatnonzero(active)
def score_array(model,a):
    if hasattr(model,'predict_proba'):
        p=model.predict_proba(a)
        if p.shape[1]==1: return np.full(len(a),float(model.classes_[0]))
        return p[:,list(model.classes_).index(1)]
    return expit(model.decision_function(a))
def score_rows(model,ii):
    return np.concatenate([score_array(model,X[ii[start:start+4096]]) for start in range(0,len(ii),4096)])
def joint_rows(model,ii):
    return np.vstack([model.predict_proba(X[ii[start:start+4096]]) for start in range(0,len(ii),4096)])
def fit_one(c,ii,target,role):
    start=time.perf_counter()
    if len(np.unique(target))<2:
        model=Pipeline([('impute',SimpleImputer()),('model',DummyClassifier(strategy='constant',constant=int(target[0])))])
    else: model=pipeline(c)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',ConvergenceWarning)
        model.fit(X[ii],target)
    fit_log.append({'config':c['name'],'role':role,'n':len(ii),'seconds':time.perf_counter()-start,
                    'warnings':'; '.join(sorted(set(str(w.message) for w in caught)))})
    return model
for c in CONFIGS:
    ii=svm_idx if c['kind']=='rbf' else search_idx
    gate_models[c['name']]=fit_one(c,ii,y[ii],'gate')
    bi=ii[(y[ii]==1)&annotated[ii]]
    assert len(bi)>0
    category_models[c['name']]={j:fit_one(c,bi,Y[bi,j],LABELS[j]) for j in active_ids}
    print('Trained',c['name'],flush=True)
# Shared trunk and sigmoid outputs: a direct eight-output neural network.
ji=search_idx[behavior_observed[search_idx]]
joint=pipeline(dict(name='joint_mlp',kind='mlp',features='all'))
joint_target=np.column_stack([y[ji],Y[ji]])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always',ConvergenceWarning)
    t=time.perf_counter(); joint.fit(X[ji],joint_target)
fit_log.append({'config':'joint_mlp','role':'joint','n':len(ji),'seconds':time.perf_counter()-t,
    'warnings':'; '.join(sorted(set(str(w.message) for w in caught)))})
training_log=pd.DataFrame(fit_log); training_log.to_csv(REPORT/'training_log.csv',index=False)
display(training_log.groupby('config').agg(seconds=('seconds','sum'),heads=('role','size')))
print('Convergence warnings:',int(training_log.warnings.ne('').sum()),'(see training_log.csv; unconverged candidates are not certified optima)')
assert len(gate_models)==len(CONFIGS)
assert joint.predict_proba(X[idx['validation'][:3]]).shape==(3,8)
print('PASS: candidate gates, binary-relevance heads and shared-output MLP fitted; test/challenge unused.')
''','5_training')
md('''## 6. Forward validation, configuration selection and thresholds

Choose the gate by validation average precision (AP), breaking ties with detection ROC AUC; select a single category configuration by macro AP on annotated validation malware with estimable labels. AP is the step-integral precision–recall summary, not trapezoidal PR AUC. Then compare this two-stage combination with the direct shared-output network using the equal-weight mean of detection AP and behavior macro AP on observed validation targets. Behavior scores for the two-stage model are gate-score × conditional category-score, used for ranking; these products are not calibrated joint probabilities.

Refit the selected architecture on the entire fitting period, leaving the configuration-validation and calibration windows separate. Freeze the gate threshold on the later calibration window to maximize recall under a **1% empirical FPR budget**. This is a study operating point, not a guaranteed deployment FPR. Choose behavior thresholds by end-to-end F1 in the same calibration window with the gate fixed. No threshold is chosen on test or challenge labels.''')
code('''def ap_safe(target,p):
    return float(average_precision_score(target,p)) if np.any(target==1) and np.any(target==0) else np.nan
def macro_ap(target,p,columns=active_ids):
    vals=[ap_safe(target[:,j],p[:,j]) for j in columns]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
def head_scores(models,ii):
    p=np.zeros((len(ii),7))
    for j,m in models.items(): p[:,j]=score_rows(m,ii)
    return p
validation_hash=sorted(idx['validation'],key=lambda i:hashlib.sha256((str(SEED)+meta.sha256.iloc[i]).encode()).digest())
vi=np.array(validation_hash[:30000]); print('Configuration validation sample:',len(vi),'of',len(idx['validation']))
vb=(y[vi]==1)&annotated[vi]; vo=behavior_observed[vi]
validation=[]
for c in CONFIGS:
    name=c['name']; g=score_rows(gate_models[name],vi); h=head_scores(category_models[name],vi)
    validation.append({'name':name,'gate_ap':ap_safe(y[vi],g),'gate_roc_auc':roc_auc_score(y[vi],g),
        'conditional_behavior_macro_ap':macro_ap(Y[vi][vb],h[vb]),
        'end_to_end_behavior_macro_ap':macro_ap(Y[vi][vo],(g[:,None]*h)[vo])})
validation=pd.DataFrame(validation).sort_values(['gate_ap','gate_roc_auc'],ascending=False)
display(validation); validation.to_csv(REPORT/'candidate_validation.csv',index=False)
best_gate=validation.iloc[0]['name']
best_heads=validation.sort_values('conditional_behavior_macro_ap',ascending=False).iloc[0]['name']
vg=score_rows(gate_models[best_gate],vi); vh=head_scores(category_models[best_heads],vi)
jp=joint_rows(joint,vi)
architecture=pd.DataFrame([
 {'architecture':'two_stage','detection_ap':ap_safe(y[vi],vg),'behavior_macro_ap':macro_ap(Y[vi][vo],(vg[:,None]*vh)[vo])},
 {'architecture':'joint_mlp','detection_ap':ap_safe(y[vi],jp[:,0]),'behavior_macro_ap':macro_ap(Y[vi][vo],jp[:,1:][vo])}])
architecture['selection_score']=(architecture.detection_ap+architecture.behavior_macro_ap)/2
winner=architecture.sort_values('selection_score',ascending=False).iloc[0].architecture
display(architecture); architecture.to_csv(REPORT/'architecture_validation.csv',index=False)
config_by_name={c['name']:c for c in CONFIGS}
fi=idx['fit']; fbi=fi[(y[fi]==1)&annotated[fi]]
# Refit the validation-selected two-stage comparator as well as the direct model
# if it wins, so gate-error propagation is measurable on the same final split.
final_gate=fit_one(config_by_name[best_gate],fi,y[fi],'final_gate')
final_heads={j:fit_one(config_by_name[best_heads],fbi,Y[fbi,j],'final_'+LABELS[j]) for j in active_ids}
final_joint=None
if winner=='joint_mlp':
    fji=fi[behavior_observed[fi]]; final_joint=clone(joint)
    final_joint.fit(X[fji],np.column_stack([y[fji],Y[fji]]))
def final_scores(ii,architecture_name):
    if architecture_name=='joint_mlp':
        p=joint_rows(final_joint,ii); return p[:,0],p[:,1:],p[:,1:]
    g=score_rows(final_gate,ii); h=head_scores(final_heads,ii)
    return g,g[:,None]*h,h
# Exact-score ties must stay together; include threshold above maximum.
def gate_threshold(target,p,budget=.01):
    from sklearn.metrics import roc_curve
    fpr,tpr,t=roc_curve(target,p,drop_intermediate=False)
    ok=np.flatnonzero(fpr<=budget)
    best=ok[np.lexsort((-t[ok],fpr[ok],-tpr[ok]))[0]]
    return float(t[best])
def f1_threshold(target,p,gated):
    if not np.any(target==1) or not np.any(target==0): return .5
    # Optimize scores after setting rejected gate rows below all finite thresholds.
    s=np.where(gated,p,-1.)
    precision,recall,t=precision_recall_curve(target,s)
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    allowed=np.flatnonzero(t>=0)
    return float(t[allowed[np.argmax(f[allowed])]]) if len(allowed) else 1.
ci=idx['calibration']; thresholds={}
for arch in dict.fromkeys(['two_stage',winner]):
    g,p,h=final_scores(ci,arch); gt=gate_threshold(y[ci],g); observed=behavior_observed[ci]
    ht=np.full(7,np.inf)
    for j in active_ids: ht[j]=f1_threshold(Y[ci][observed,j],p[observed,j],g[observed]>=gt)
    thresholds[arch]={'gate':gt,'behavior':ht}
    fp=((g>=gt)&(y[ci]==0)).sum()/max(1,(y[ci]==0).sum())
    print(arch,'calibration FPR',fp,'threshold',gt)
    assert fp<=.01+1e-12
selection={'winner':winner,'best_gate':best_gate,'best_heads':best_heads,'active_labels':[LABELS[j] for j in active_ids],
 'thresholds':{a:{'gate':v['gate'],'behavior':[float(x) if np.isfinite(x) else None for x in v['behavior']]} for a,v in thresholds.items()},
 'candidate_sample_n':len(search_idx),'configuration_validation_n':len(vi),'fitting_n':len(fi),
 'source_records_sha256':fingerprint.hexdigest(),'feature_cache_key':key,'fitting_file_types':sorted(meta.iloc[fi].file_type.unique().tolist()),'selection_rule':'mean validation detection AP and observed-target behavior macro AP'}
(ART/'selection.json').write_text(json.dumps(selection,indent=2))
joblib.dump({'gate':final_gate,'heads':final_heads,'joint':final_joint,'thresholds':thresholds,
    'labels':LABELS,'aliases':ALIASES,'active':active,'feature_groups':groups,'selection':selection},ART/'selected_models.joblib')
pd.DataFrame(fit_log).to_csv(REPORT/'training_log.csv',index=False)
print('Selected:',winner,'; gate:',best_gate,'; behavior:',best_heads)
print('PASS: configuration selected on earlier validation; thresholds frozen on later calibration; final models saved.')
''','6_validation')
md('''## 7. Final evaluation: official test and challenge

Use frozen models and thresholds exactly once. Report detection precision, recall, F1, AP, ROC AUC, FPR, false positives and missed malware. Report behavior micro/macro F1, AP, subset accuracy and Hamming loss over estimable labels, per-category support and coverage. Category evaluation uses benign plus annotated-malicious records; also report results conditional on annotated malware. Untagged malware still contributes to detection metrics, never to assumed-negative behavior metrics.

Challenge is curated malicious-only data: recall/miss rate are meaningful; FPR and detection ROC/AP as a discrimination comparison are undefined without benign records. File-format shift limits conclusions when fitting only APK/PDF/Win64. Separate oracle-gated category results from actual-gated results to quantify inherited detection errors.

Uncertainty uses a seeded bootstrap of chronological week clusters (200 replicates) for gate recall/FPR and end-to-end micro-F1. These intervals reflect week-to-week variability under this observed dataset, not a guarantee about future malware. Undefined replicates are skipped. Also inspect performance by file type and malware family to reveal concentration and drift.''')
code('''from sklearn.metrics import multilabel_confusion_matrix
summaries=[]; per_category=[]; propagation=[]; group_rows=[]; intervals=[]; predictions={}
def gate_metrics(target,p,pred):
    tn,fp,fn,tp=confusion_matrix(target,pred,labels=[0,1]).ravel()
    both=len(np.unique(target))==2
    return {'n':len(target),'precision':precision_score(target,pred,zero_division=0),
      'recall':recall_score(target,pred,zero_division=0),'f1':f1_score(target,pred,zero_division=0),
      'ap':ap_safe(target,p),'roc_auc':roc_auc_score(target,p) if both else np.nan,
      'fpr':fp/(fp+tn) if fp+tn else np.nan,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
def behavior_metrics(target,p,pred):
    t=target[:,active_ids]; pp=p[:,active_ids]; pr=pred[:,active_ids]
    valid=[k for k in range(t.shape[1]) if len(np.unique(t[:,k]))==2]
    return {'behavior_n':len(t),'behavior_micro_f1':f1_score(t,pr,average='micro',zero_division=0),
      'behavior_macro_f1':f1_score(t[:,valid],pr[:,valid],average='macro',zero_division=0) if valid else np.nan,
      'behavior_macro_ap':macro_ap(target,p),'behavior_micro_ap':ap_safe(t.ravel(),pp.ravel()),
      'subset_accuracy':accuracy_score(t,pr),'hamming_loss':hamming_loss(t,pr),
      'macro_evaluable_labels':len(valid)}
for split in ['test','challenge']:
    ii=idx[split]; observed=behavior_observed[ii]
    print('Evaluating',split,'file types:',meta.iloc[ii].file_type.value_counts().to_dict())
    for arch in dict.fromkeys(['two_stage',winner]):
        g,p,h=final_scores(ii,arch); gt=thresholds[arch]['gate']; ht=thresholds[arch]['behavior']
        gate=g>=gt; pred=(p>=ht)&gate[:,None]; pred[:,~active]=False
        row={'split':split,'architecture':arch,**gate_metrics(y[ii],g,gate),
             **behavior_metrics(Y[ii][observed],p[observed],pred[observed]),
             'observed_behavior_coverage':float(observed.mean())}
        summaries.append(row)
        predictions[(split,arch)]=(ii,g,p,gate,pred)
        for j,label in enumerate(LABELS):
            t=Y[ii][observed,j]; pp=p[observed,j]; pr=pred[observed,j]
            per_category.append({'split':split,'architecture':arch,'category':label,'estimable':bool(active[j]),
                'n_observed':len(t),'positives':int(t.sum()),'precision':precision_score(t,pr,zero_division=0) if active[j] else np.nan,
                'recall':recall_score(t,pr,zero_division=0) if active[j] and t.sum() else np.nan,
                'f1':f1_score(t,pr,zero_division=0) if active[j] and t.sum() else np.nan,
                'ap':ap_safe(t,pp) if active[j] else np.nan})
        for ft in meta.iloc[ii].file_type.unique():
            mask=meta.iloc[ii].file_type.eq(ft).to_numpy()
            group_rows.append({'split':split,'architecture':arch,'file_type':ft,**gate_metrics(y[ii][mask],g[mask],gate[mask])})
        if arch=='two_stage':
            mal=(y[ii]==1)&annotated[ii]
            # Same conditional head decisions with and without actual gate.
            # Use the frozen gate threshold as a fixed reference multiplier for
            # oracle scores, avoiding impossible category recovery when gate=0.
            oracle_scores=h*max(gt,1e-12)
            oracle=(oracle_scores>=ht); oracle[:,~active]=False
            oracle_f1=f1_score(Y[ii][mal][:,active_ids],oracle[mal][:,active_ids],average='micro',zero_division=0)
            actual_f1=f1_score(Y[ii][mal][:,active_ids],pred[mal][:,active_ids],average='micro',zero_division=0)
            # Pure gating ablation holds each observed score fixed and removes only
            # the gate veto. This isolates the hard-gate loss from ranking effects.
            ungated=(p>=ht); ungated[:,~active]=False
            ablated=f1_score(Y[ii][mal][:,active_ids],ungated[mal][:,active_ids],average='micro',zero_division=0)
            propagation.append({'split':split,'annotated_malware_n':int(mal.sum()),
               'actual_micro_f1':actual_f1,'no_gate_veto_micro_f1':ablated,
               'micro_f1_loss_from_hard_gate':ablated-actual_f1,
               'positive_labels_lost_to_gate':int(Y[ii][mal & ~gate][:,active_ids].sum()),
               'all_positive_labels':int(Y[ii][mal][:,active_ids].sum())})
        # Week-cluster bootstrap: retain all records in a resampled week.
        weeks=meta.iloc[ii].date.dt.strftime('%G-%V').to_numpy(); unique=np.unique(weeks)
        week_members=[np.flatnonzero(weeks==w) for w in unique]; rng=np.random.default_rng(SEED)
        boot={'recall':[],'fpr':[],'behavior_micro_f1':[]}
        for b in range(200):
            take=np.concatenate([week_members[k] for k in rng.integers(0,len(unique),len(unique))])
            yt=y[ii][take]; gp=gate[take]; pos=yt==1; neg=yt==0
            boot['recall'].append(float(gp[pos].mean()) if pos.any() else np.nan)
            boot['fpr'].append(float(gp[neg].mean()) if neg.any() else np.nan)
            obs=observed[take]; ti=take[obs]
            boot['behavior_micro_f1'].append(f1_score(Y[ii][ti][:,active_ids],pred[ti][:,active_ids],average='micro',zero_division=0))
        for metric,values in boot.items():
            a=np.asarray(values); a=a[np.isfinite(a)]
            intervals.append({'split':split,'architecture':arch,'metric':metric,'weeks':len(unique),'valid_replicates':len(a),
                'lower95':float(np.quantile(a,.025)) if len(a) else np.nan,'upper95':float(np.quantile(a,.975)) if len(a) else np.nan})
summary=pd.DataFrame(summaries); category=pd.DataFrame(per_category); drift=pd.DataFrame(group_rows)
summary.to_csv(REPORT/'final_metrics.csv',index=False); category.to_csv(REPORT/'category_metrics.csv',index=False)
drift.to_csv(REPORT/'filetype_metrics.csv',index=False); pd.DataFrame(propagation).to_csv(REPORT/'gate_error_propagation.csv',index=False)
pd.DataFrame(intervals).to_csv(REPORT/'bootstrap_intervals.csv',index=False)
display(summary); display(category); display(pd.DataFrame(propagation)); display(pd.DataFrame(intervals))
# Descriptive family errors; this is an error slice, not a family-disjoint test.
ii,g,p,gate,pred=predictions[('test',winner)]
family_errors=meta.iloc[ii].assign(detected=gate).query('malicious == 1').groupby('family').agg(n=('sha256','size'),recall=('detected','mean')).sort_values('n',ascending=False)
family_errors.to_csv(REPORT/'test_family_errors.csv'); display(family_errors.head(15))
fig,axes=plt.subplots(1,2,figsize=(12,4))
pr,rc,_=precision_recall_curve(y[ii],g); axes[0].plot(rc,pr); axes[0].set(xlabel='Recall',ylabel='Precision',title='Held-out test detection PR curve')
m=category.query('split == "test" and architecture == @winner').set_index('category')
m.f1.plot.barh(ax=axes[1],title='Test behavior F1 (supported categories)'); plt.tight_layout()
fig.savefig(REPORT/'figures/02_test_results.png',dpi=150,bbox_inches='tight'); plt.show()
assert summary.query('split == "challenge"').fpr.isna().all()
assert summary.query('split == "challenge"').roc_auc.isna().all()
assert all(not np.any(prd[:,~active]) for _,_,_,_,prd in predictions.values())
print('PASS: frozen held-out evaluation; undefined malicious-only metrics retained as NaN; results exported.')
''','7_evaluation')
md('''## 8. Interpretation, reproducibility and limitations

Read results in context: the file-type subset and incomplete antivirus annotations constrain the task; a high micro score can hide poor minority-category recall. Validation chooses the architecture, while test only estimates that frozen choice. Challenge recall should be interpreted alongside format-specific recall. The selected 1% calibration FPR budget may be exceeded after temporal drift. A future study should add PE fitting data, inspect alias quality against expert-reviewed annotations, repeat multiple seeds, test fuzzy TLSH distances/family-disjoint splits, and consider a masked-loss multi-head neural network with genuine missing-label handling.

The model artifact stores all fitted preprocessing, heads, thresholds, aliases and selection metadata. Load joblib files only from trusted sources. Static feature inputs can be supplied without executing a suspicious binary. Re-running this notebook reproduces selection and evaluation; vector-cache verification avoids repeated extraction.''')
code('''test_row=summary.query('split == "test" and architecture == @winner').iloc[0]
challenge_row=summary.query('split == "challenge" and architecture == @winner').iloc[0]
unsupported=[LABELS[j] for j in range(7) if not active[j]]
text=f"""# Experimental findings

Selected architecture: **{winner}**. The two-stage comparator used **{best_gate}** for detection and **{best_heads}** for categories.

This run retained {len(meta):,} metadata records, with {len(idx['fit']):,} eligible fitting records, {len(idx['validation']):,} configuration-validation records, {len(idx['calibration']):,} threshold-calibration records, {len(idx['test']):,} test records and {len(idx['challenge']):,} challenge records. Configuration search used {len(search_idx):,} hash-selected fitting records; the RBF SVM used {len(svm_idx):,}.

At the frozen operating point, test detection precision was {test_row.precision:.4f}, recall {test_row.recall:.4f}, F1 {test_row.f1:.4f}, AP {test_row.ap:.4f}, and FPR {test_row.fpr:.4f}. There were {int(test_row.fp):,} false alarms and {int(test_row.fn):,} missed malware samples. Behavior micro-F1 was {test_row.behavior_micro_f1:.4f} and macro-AP {test_row.behavior_macro_ap:.4f} on observed targets; behavior annotation coverage was {test_row.observed_behavior_coverage:.1%}.

Challenge detection recall was {challenge_row.recall:.4f}, with {int(challenge_row.fn):,} missed files. Challenge FPR and discrimination AUC are undefined because all records are malicious. Challenge behavior annotation coverage was {challenge_row.observed_behavior_coverage:.1%}.

Unsupported categories: {', '.join(unsupported) if unsupported else 'none at fitting eligibility threshold'}. Review per-category support in `category_metrics.csv` before drawing conclusions about all seven roles.

Limitations: APK/PDF/Win64-only fitting, challenge format shift, weak/noisy labels, incomplete behavior annotation, exact-match rather than fuzzy similarity purging, one seed, bounded hyperparameter search, and shallow/unconverged neural candidates where recorded. These are subset results, not full EMBER2024 benchmark claims. Scores are not calibrated probabilities.

See `candidate_validation.csv`, `architecture_validation.csv`, `training_log.csv`, `final_metrics.csv`, `category_metrics.csv`, `filetype_metrics.csv`, `bootstrap_intervals.csv`, and `gate_error_propagation.csv` for auditable evidence. Package versions, source identities, feature groups and final selection are in `artifacts/`.
"""
(REPORT/'findings.md').write_text(text)
display(Markdown(text))
# Small inference contract check: every raw feature group is static, every output
# has a fixed label meaning, unavailable labels are never presented as predictions.
sample_g,sample_p,_=final_scores(idx['test'][:5],winner)
assert sample_p.shape==(5,7) and np.isfinite(sample_p).all()
print('PASS: exported report, trained model bundle, seven-output inference contract.')
''','8_interpretation')
for cell in cells:
    cell.source=cell.source.replace('2506.01538','2506.05074')
    if cell.metadata.get('stage')=='1_data_audit':
        cell.source=cell.source.replace("fingerprint.update((d['sha256']+'|'+str(d.get('first_submission_date'))+'|'+official+'\\n').encode())", "fingerprint.update(line.encode('utf-8'))")
        cell.source=cell.source.replace('ordered_metadata_sha256','source_records_sha256')
        cell.source=cell.source.replace("    aa=set(meta.loc[meta.split.eq(a)&valid_tlsh.reindex(meta.index,fill_value=False),'tlsh'].dropna()) if False else None\n",'')
        cell.source=cell.source.replace("display(pd.Series(behavior_counts,name='training tag count')", "behavior_counts=Counter(t for tags in meta.loc[meta.official.eq('train'),'tags'] for t in tags)\ndisplay(pd.Series(behavior_counts,name='deduplicated training tag count')")
    if cell.metadata.get('stage')=='4_normalization':
        cell.source=Path(__file__).with_name('feature_statistics.py').read_text()+'\n'+cell.source
        cell.source=cell.source.replace('def column_subset(a,columns): return a[:,columns]', 'from multi_label_static_analysis.research_utils import column_subset')
nb.cells=cells
out.parent.mkdir(parents=True,exist_ok=True)
nbf.write(nb,out)
print(out)
