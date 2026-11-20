"""Build the primary data/feature analysis notebook. No classifier training."""
from pathlib import Path
import argparse
import nbformat as nbf
from multi_label_static_analysis.config import project_paths
P=project_paths()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=P.notebook)
parser.add_argument('--force',action='store_true')
args=parser.parse_args(); out=args.output.resolve()
if out.exists() and not args.force: parser.error('Output exists; choose --output or explicitly use --force.')
nb=nbf.v4.new_notebook(); nb.metadata.kernelspec={'name':'python3','display_name':'Python (uv analysis)','language':'python'}
cells=[]
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s,stage):
 c=nbf.v4.new_code_cell(s); c.metadata['stage']=stage; cells.append(c)
md('''# EMBER2024: primary data and feature analysis

**Current milestone: understand and prepare the data.** This notebook performs ingestion, data-quality and label audits, leakage-aware split construction, static vectorization, descriptive statistics, normalization, filter-based feature selection, PCA, and visualization. It does **not** train, tune, load, or evaluate a predictive classifier. Feature selectors and PCA are exploratory transformations; no fitted objects are serialized.

Analyze the feature records actually present in the input directory. The current inventory contains Win64 train/test data and challenge records; APK/PDF fitting files are absent. Do not silently assume the earlier mixed-format population is still present. Preserve official test/challenge boundaries; use only the fitting period for feature ranking, normalization statistics and PCA. Validation features are inspected for descriptive drift. Historical held-out scores have already been inspected in earlier work and are retained as references, not recomputed results or an untouched future experiment.

The seven proposed categories are weak antivirus-derived proxies, not verified capabilities. Missing malware behavior annotations are unknown, not confirmed negatives. The next model-building milestone is intentionally left open until we review these findings.''')
code('''from pathlib import Path
import json, hashlib, platform, warnings
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from threadpoolctl import threadpool_limits
thread_limits=threadpool_limits(limits=4)
from multi_label_static_analysis.config import project_paths
from multi_label_static_analysis.data_analysis import (stream_metadata, common_extractors, vectorize_record,
    feature_dictionary, binary_anova, verify_score_reference, assert_no_models)
P=project_paths(); DATA=P.data; REPORT=P.reports; ART=P.artifacts
REPORT.mkdir(parents=True,exist_ok=True); (REPORT/'figures').mkdir(exist_ok=True)
SEED=603
rng=np.random.default_rng(SEED)
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
def savefig(name):
    plt.tight_layout(); plt.savefig(REPORT/'figures'/name,dpi=150,bbox_inches='tight'); plt.show()
def table(frame,name):
    frame.to_csv(REPORT/name,index=False); display(frame)
assert_no_models(ART)
score_reference=verify_score_reference(P.historical)
paths=sorted(DATA.glob('*.jsonl')); assert paths, 'Extract feature JSONL files first.'
print('Data files:',len(paths),'source GB:',round(sum(p.stat().st_size for p in paths)/1e9,2))
print('Current environment:',platform.python_version())
print('Historical score files protected by SHA256:',len(score_reference['files']))
print('PASS: inputs present, no trained models retained, historical score checksums verified.')
''','1_scope_inventory')
md('''## 2. Streaming ingestion and raw schema audit

Read every JSONL record once, hash its raw UTF-8 text, and retain metadata rather than full raw features in RAM. Audit required feature groups, histogram lengths, invalid JSON and missing fields. Source filenames identify the official partition; timestamps determine chronological windows. Filename week strings do not determine record chronology. File size, entropy and family confidence are descriptive metadata; only explicitly allowlisted static groups later enter X.

Stop on malformed records or missing feature groups instead of silently dropping an unknown population. Duplicate records remain in this raw audit so their extent and contradictory annotations can be quantified.''')
code('''rows,source_profile,source_sha,schema_issues,parse_errors=stream_metadata(paths)
raw=pd.DataFrame(rows); del rows
raw['date']=pd.to_datetime(raw.timestamp,unit='s',utc=True,errors='coerce')
raw['family_confidence']=pd.to_numeric(raw.family_confidence,errors='coerce')
raw['file_size']=pd.to_numeric(raw.file_size,errors='coerce'); raw['byte_entropy']=pd.to_numeric(raw.byte_entropy,errors='coerce')
inventory=pd.DataFrame(source_profile)
inventory.assign(file_types=inventory.file_types.map(json.dumps)).to_csv(REPORT/'source_inventory.csv',index=False)
quality={'raw_rows':len(raw),'files':len(paths),'source_records_sha256':source_sha,
 'parse_errors':len(parse_errors),'schema_issues':schema_issues,
 'missing_dates':int(raw.date.isna().sum()),'missing_identity':int(raw.sha256.isna().sum()),
 'invalid_binary_labels':int((~raw.malicious.isin([0,1])).sum()),
 'missing_family':int(raw.family.isna().sum()),'empty_behavior_lists':int(raw.tags.map(len).eq(0).sum())}
(REPORT/'ingestion_quality.json').write_text(json.dumps(quality,indent=2))
display(pd.Series(quality,dtype=object))
table(raw.groupby(['official','file_type']).size().rename('raw_rows').reset_index(),'raw_population.csv')
assert not parse_errors and not schema_issues, 'Review raw schema errors before vectorization.'
assert raw.date.notna().all() and raw.malicious.isin([0,1]).all()
print('PASS: every raw record ingested; schema, labels and timestamps audited.')
''','2_ingestion')
md('''## 3. Duplicates, annotation conflicts and chronological partitions

Count repeated SHA identities before deduplication. Repeated feature records must not be treated as independent observations. Report disagreements in binary labels and behavior tags separately. Retain the earliest timestamp/source representative, matching the historical cache's identity policy; report any contradictory labels as a data-quality issue rather than claiming certainty.

Split official training into its first 40 weeks for fitting, next 6 weeks for configuration validation, and last 6 for later calibration. Retain official test and challenge partitions. For valid exact TLSH matches, keep the group only in its earliest partition and purge later members. This controls exact-match leakage; it is not full fuzzy-distance or family-disjoint clustering.''')
code('''duplicate_counts=raw.groupby('sha256',sort=False).size()
conflicts=raw.groupby('sha256',sort=False).agg(binary_versions=('malicious','nunique'),behavior_versions=('tags','nunique'),official_partitions=('official','nunique'))
conflict_summary={'repeated_identities':int((duplicate_counts>1).sum()),'repeated_rows':int(raw.sha256.duplicated().sum()),
 'binary_conflict_identities':int(conflicts.binary_versions.gt(1).sum()),'behavior_conflict_identities':int(conflicts.behavior_versions.gt(1).sum()),
 'cross_official_identity_groups':int(conflicts.official_partitions.gt(1).sum())}
conflicts.query('binary_versions > 1 or behavior_versions > 1').head(100).reset_index().to_csv(REPORT/'duplicate_conflict_examples.csv',index=False)
(REPORT/'duplicate_audit.json').write_text(json.dumps(conflict_summary,indent=2)); display(pd.Series(conflict_summary))
train_end=raw.loc[raw.official.eq('train'),'date'].max().normalize()+pd.Timedelta(days=1)
val_start=train_end-pd.Timedelta(weeks=12); cal_start=train_end-pd.Timedelta(weeks=6)
raw['split']=raw.official
raw.loc[raw.official.eq('train')&raw.date.ge(val_start),'split']='validation'
raw.loc[raw.official.eq('train')&raw.date.ge(cal_start),'split']='calibration'
raw.loc[raw.split.eq('train'),'split']='fit'
meta=raw.sort_values(['date','sha256','source','line']).drop_duplicates('sha256',keep='first').copy()
del raw,conflicts,duplicate_counts
order={'fit':0,'validation':1,'calibration':2,'test':3,'challenge':4}
meta['priority']=meta.split.map(order)
valid_tlsh=meta.tlsh.notna()&~meta.tlsh.isin(['','TNULL','NULL','-'])
first_group=meta.loc[valid_tlsh].groupby('tlsh').priority.min()
later=valid_tlsh&meta.priority.gt(meta.tlsh.map(first_group))
tlsh_purged=int(later.sum()); meta=meta.loc[~later].reset_index(drop=True); meta['row_id']=np.arange(len(meta))
assert meta.sha256.is_unique
for a,b in [('fit','validation'),('validation','calibration'),('calibration','test')]:
    assert meta.loc[meta.split.eq(a),'date'].max()<meta.loc[meta.split.eq(b),'date'].min()
v=meta.tlsh.notna()&~meta.tlsh.isin(['','TNULL','NULL','-'])
assert meta.loc[v].groupby('tlsh').split.nunique().max()==1
split_summary=meta.groupby(['split','file_type']).agg(n=('sha256','size'),start=('date','min'),end=('date','max')).reset_index()
split_summary.to_csv(REPORT/'chronological_population.csv',index=False); display(split_summary)
print('Exact TLSH cross-partition purges:',tlsh_purged)
fig,axes=plt.subplots(1,2,figsize=(13,4))
meta.groupby(['split','file_type']).size().unstack(fill_value=0).reindex(order).plot.bar(stacked=True,ax=axes[0],rot=20,title='Retained metadata by partition and format')
weekly=meta[meta.official.ne('challenge')].assign(week=meta.date.dt.strftime('%G-%V')).groupby(['week','file_type']).size().unstack(fill_value=0)
weekly.plot.area(ax=axes[1],title='Record timeline (challenge excluded)',legend=False); axes[1].set_ylabel('Records')
savefig('01_population_and_timeline.png')
print('PASS: unique identities and time-forward partitions verified; exact TLSH overlaps purged.')
''','3_identity_time_audit')
md('''## 4. Label quality, imbalance and overlapping categories

Map only explicit behavior aliases to the seven proposal roles. An empty malware behavior list is missing annotation. Other observed behavior tags are imperfect proxy negatives for a particular mapped role. Family/confidence fields are audited but excluded from input features. Compute support using the fitting period only; require at least 20 positives and negatives for a candidate role. An unsupported category is not evidence that the malware behavior is absent.

Show format-specific class balance, annotation coverage, dominant tags/families, role co-occurrence and label cardinality. These establish what a later study can honestly evaluate before any classifier is chosen.''')
code('''ALIASES={'Ransomware':{'ransomware','ransom'},'Infostealer':{'infostealer','stealer','pws','passwordstealer'},
 'Stager':{'downloader','dropper','loader'},'Remote Access Trojan':{'backdoor','rat','remoteaccess'},
 'Botnet Agent':{'botnet','bot'},'Banking Trojan':{'banker','banking','bankingtrojan'},
 'Cryptominer':{'miner','coinminer','cryptominer','cryptocurrencyminer'}}
LABELS=list(ALIASES)
Y=np.array([[int(bool(set(tags)&ALIASES[label])) for label in LABELS] for tags in meta.tags],dtype=np.int8)
y=meta.malicious.to_numpy(dtype=np.int8); annotated=meta.tags.map(len).gt(0).to_numpy()
fit=meta.split.eq('fit').to_numpy(); observed=(y==0)|annotated; mal_annotated=fit&(y==1)&annotated
support=pd.DataFrame({'category':LABELS,'fit_positive':Y[fit&observed].sum(axis=0),
 'conditional_negative':int(mal_annotated.sum())-Y[mal_annotated].sum(axis=0)})
supported=(support.fit_positive.to_numpy()>=20)&(support.conditional_negative.to_numpy()>=20)
support['supported_for_future_study']=supported; table(support,'label_support.csv')
coverage=meta[fit].assign(annotated=annotated[fit]).groupby(['file_type','malicious']).agg(n=('sha256','size'),annotation_rate=('annotated','mean')).reset_index()
table(coverage,'fitting_annotation_coverage.csv')
tag_counts=Counter(t for tags in meta.loc[fit,'tags'] for t in tags)
tag_table=pd.DataFrame(tag_counts.most_common(),columns=['tag','count']); tag_table.to_csv(REPORT/'fitting_behavior_tags.csv',index=False)
family_table=meta.loc[fit,'family'].value_counts(dropna=False).rename_axis('family').reset_index(name='n'); family_table.to_csv(REPORT/'fitting_families.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(15,10))
meta.loc[fit].groupby(['file_type','malicious']).size().unstack(fill_value=0).plot.bar(ax=axes[0,0],rot=0,title='Fitting binary class balance')
coverage[coverage.malicious.eq(1)].set_index('file_type').annotation_rate.plot.bar(ax=axes[0,1],rot=0,ylim=(0,1),title='Malware behavior annotation coverage')
co=Y[mal_annotated].astype(np.int64).T@Y[mal_annotated].astype(np.int64)
im=axes[1,0].imshow(co,cmap='Blues'); axes[1,0].set_xticks(range(7),LABELS,rotation=65,ha='right'); axes[1,0].set_yticks(range(7),LABELS); axes[1,0].set_title('Fitting role co-occurrence'); fig.colorbar(im,ax=axes[1,0])
axes[1,1].hist(Y[mal_annotated].sum(axis=1),bins=np.arange(9)-.5,rwidth=.85); axes[1,1].set(title='Mapped-role cardinality (annotated malware)',xlabel='Number of mapped roles',ylabel='Records')
savefig('02_labels_and_coverage.png')
fig,axes=plt.subplots(1,2,figsize=(14,5))
tag_table.head(15).set_index('tag')['count'].sort_values().plot.barh(ax=axes[0],title='Most frequent fitting behavior tags')
meta.loc[fit,'family_confidence'].dropna().hist(bins=40,ax=axes[1]); axes[1].set(title='Family confidence (metadata only)',xlabel='Confidence',ylabel='Records')
savefig('03_tag_and_annotation_quality.png')
print('Unannotated fitting malware:',int((fit&(y==1)&~annotated).sum()))
print('PASS: role support and label-quality rules use fitting-period metadata only.')
''','4_label_analysis')
md('''## 5. Static vectorization, semantic dictionary and cache validation

Convert only `general`, `histogram`, `byteentropy`, and `strings` through thrember. Identity hashes, dates, AV labels, family/confidence and behavior tags are explicitly outside this allowlist. These four groups yield 696 shared inputs. Win64 has additional PE fields, but the present cross-format analysis intentionally examines the shared representation.

Reuse the prior input matrix only when the complete source-text digest, canonical source/row identities and feature-extractor source hash match. Independently regenerate representative rows from every file type to verify cache alignment. Otherwise stream a fresh float32 memory-mapped matrix. Histogram groups are frequencies; counts, entropy and categorical start bytes retain distinct semantics. Purge identical feature vectors from later partitions conservatively, without using labels.''')
code('''from multi_label_static_analysis.thrember import features as feature_module
extractors=common_extractors(); names=feature_dictionary(extractors)
groups={}; offset=0
for ex in extractors: groups[ex.name]=list(range(offset,offset+ex.dim)); offset+=ex.dim
D=offset; assert D==696 and len(set(names))==D
feature_source_sha=hashlib.sha256(Path(feature_module.__file__).read_bytes()).hexdigest()
key=hashlib.sha256((source_sha+str(meta[['source','line']].values.tolist())+feature_source_sha).encode()).hexdigest()
cache=ART/'static_features.npy'; keyfile=ART/'feature_cache_key.txt'
if cache.exists() and keyfile.exists() and keyfile.read_text()==key:
    X=np.load(cache,mmap_mode='r'); print('Reused source-verified input matrix:',X.shape)
else:
    X=np.lib.format.open_memmap(cache,mode='w+',dtype='float32',shape=(len(meta),D))
    lookup={(r.source,int(r.line)):int(r.row_id) for r in meta.itertuples()}; n=0
    for path in paths:
        with path.open() as stream:
            for line_no,line in enumerate(stream):
                i=lookup.get((path.name,line_no))
                if i is None: continue
                X[i]=vectorize_record(json.loads(line),extractors); n+=1
    assert n==len(meta); X.flush(); keyfile.write_text(key); del lookup
    print('Stream-vectorized',n,'records.')
assert X.shape==(len(meta),D)
# Select a few representative source rows per format, then stream only relevant files.
checks=[]
for ft in sorted(meta.file_type.unique()):
    part=meta[meta.file_type.eq(ft)]; source=part.source.iloc[0]
    checks.extend(part[part.source.eq(source)].head(6)[['source','line','row_id']].itertuples(index=False,name=None))
check_map={}
for source,line,i in checks: check_map.setdefault(source,{})[int(line)]=int(i)
checked=0
for source,wanted in check_map.items():
    with (DATA/source).open() as stream:
        for line_no,line in enumerate(stream):
            if line_no in wanted:
                expected=vectorize_record(json.loads(line),extractors)
                assert np.allclose(expected,X[wanted[line_no]],equal_nan=True)
                checked+=1
            if line_no>=max(wanted): break
print('Raw-to-vector independent row checks:',checked)
nonfinite=sum(int((~np.isfinite(X[start:start+4096])).sum()) for start in range(0,len(X),4096))
vector_groups={}; keep=np.ones(len(meta),dtype=bool)
for i in np.argsort(meta.priority.to_numpy(),kind='stable'):
    signature=hashlib.blake2b(np.nan_to_num(X[i],nan=-1).tobytes(),digest_size=16).digest()
    previous=vector_groups.setdefault(signature,meta.split.iloc[i])
    if previous!=meta.split.iloc[i]: keep[i]=False
idx={s:np.flatnonzero(meta.split.eq(s).to_numpy()&keep) for s in order}
del vector_groups
assert all(len(idx[s]) for s in order)
dictionary=pd.DataFrame({'index':range(D),'feature':names,'group':[n.split('.')[0] for n in names]})
dictionary['kind']='continuous'
dictionary.loc[dictionary.feature.isin(['general.is_pe']+[f'general.start_byte_{i}' for i in range(4)]),'kind']='categorical'
dictionary.to_csv(REPORT/'feature_dictionary.csv',index=False)
counts=meta.loc[keep].groupby(['split','file_type']).size().rename('n').reset_index(); table(counts,'eligible_partition_counts.csv')
vector_quality={'rows':len(meta),'features':D,'nonfinite_cells':nonfinite,'vector_purges':int((~keep).sum()),
 'independent_row_checks':checked,'feature_cache_key':key,'extractor_sha256':feature_source_sha}
(REPORT/'vectorization_quality.json').write_text(json.dumps(vector_quality,indent=2))
probe=idx['fit'][:10000]
for group in ['histogram','byteentropy']:
    total=np.nansum(X[np.ix_(probe,groups[group])],axis=1)
    assert np.allclose(total,1,atol=1e-5)
assert_no_models(ART)
print('PASS: static-only 696-column inputs, semantic names, cache alignment and histogram normalization checked.')
''','5_vectorization')
md('''## 6. Exact fitting-feature statistics and distributions

Compute missingness, zero prevalence, mean, standard deviation, percentiles, skewness and extreme-tail rates on **all eligible fitting records**, using 16-column blocks to bound RAM. Constants are identified on fitting data only. Inspect distributions by format because feature range and meaning differ across APK/PDF/Win64. Long-tailed count features need different treatment from normalized frequencies or categorical bytes; a single blanket transformation is not justified.''')
code('''stat_records=[]
for start in range(0,D,16):
    columns=np.arange(start,min(start+16,D)); a=np.asarray(X[np.ix_(idx['fit'],columns)],dtype=np.float64)
    mean=np.nanmean(a,axis=0); std=np.nanstd(a,axis=0); median=np.nanmedian(a,axis=0)
    p01,p25,p75,p99=np.nanpercentile(a,[1,25,75,99],axis=0)
    with np.errstate(divide='ignore',invalid='ignore'):
        skew=np.nanmean(((a-mean)/np.where(std>0,std,1))**3,axis=0)
    for k,j in enumerate(columns):
        stat_records.append({'index':int(j),'feature':names[j],'group':names[j].split('.')[0],
          'missing_rate':float(np.isnan(a[:,k]).mean()),'zero_rate':float((a[:,k]==0).mean()),
          'mean':float(mean[k]),'std':float(std[k]),'median':float(median[k]),'p01':float(p01[k]),
          'p25':float(p25[k]),'p75':float(p75[k]),'p99':float(p99[k]),'skewness':float(skew[k]),
          'above_tukey_fence_rate':float((a[:,k]>p75[k]+1.5*(p75[k]-p25[k])).mean())})
    del a
stats=pd.DataFrame(stat_records); stats['constant']=stats['std'].eq(0)
stats.to_csv(REPORT/'fitting_feature_statistics.csv',index=False)
variable=np.flatnonzero(~stats.constant.to_numpy())
table(stats.groupby('group').agg(dimensions=('index','size'),constants=('constant','sum'),mean_zero_rate=('zero_rate','mean')).reset_index(),'feature_group_summary.csv')
display(stats.sort_values('skewness',ascending=False).head(12)[['feature','median','p99','skewness','zero_rate']])
fig,axes=plt.subplots(2,2,figsize=(13,9))
fitmeta=meta.iloc[idx['fit']]
for ft in sorted(fitmeta.file_type.unique()):
    take=idx['fit'][fitmeta.file_type.eq(ft).to_numpy()]
    axes[0,0].hist(np.log10(1+X[take,0]),bins=60,density=True,alpha=.35,label=ft)
    axes[0,1].hist(X[take,1],bins=60,density=True,alpha=.35,label=ft)
    axes[1,0].hist(np.log10(1+X[take,groups['strings'][0]]),bins=60,density=True,alpha=.35,label=ft)
axes[0,0].set(xlabel='log10(1 + bytes)',title='File size by fitting format')
axes[0,1].set(xlabel='Byte entropy',title='Entropy by fitting format')
axes[1,0].set(xlabel='log10(1 + string count)',title='String count by fitting format')
stats.groupby('group').zero_rate.mean().plot.bar(ax=axes[1,1],rot=20,title='Mean zero prevalence by group',ylim=(0,1))
for ax in axes.flat[:3]: ax.legend()
savefig('04_feature_distributions.png')
print('Variable fitting columns:',len(variable),'constant columns:',int(stats.constant.sum()))
print('PASS: exact full-fitting feature statistics computed without materializing the complete matrix.')
''','6_feature_statistics')
md('''## 7. Normalization and redundancy

Use fitting medians for missing values and fitting means/standard deviations for scaling. The exact statistics above cover the full fitting period; only displayed correlations and scatter points use a deterministic, label-independent sample capped at 20,000 rows. Constants are excluded from correlation/PCA. Scaling addresses units but not all outliers, so compare count distributions before/after log1p as a diagnostic, not an automatically selected preprocessing rule.

Correlation indicates redundancy, not causal equivalence. Pearson correlation can be outlier-sensitive; report the high-correlation pairs explicitly rather than assuming all correlated inputs are disposable for every future target.''')
code('''hash_order=sorted(idx['fit'],key=lambda i:hashlib.sha256((str(SEED)+meta.sha256.iloc[i]).encode()).digest())
sample_idx=np.array(hash_order[:20000]); medians=stats['median'].to_numpy(); means=stats['mean'].to_numpy(); scales=stats['std'].to_numpy()
def standardized(ii):
    a=np.asarray(X[ii],dtype=np.float64)
    a=np.where(np.isfinite(a),a,medians)
    return ((a-means)/np.where(scales>0,scales,1))[:,variable]
Z=standardized(sample_idx); assert np.isfinite(Z).all()
correlation=np.corrcoef(Z,rowvar=False)
upper=np.triu(np.abs(correlation),k=1); left,right=np.where(upper>=.98)
pairs=pd.DataFrame({'feature_a':[names[variable[i]] for i in left],'feature_b':[names[variable[j]] for j in right],'abs_correlation':upper[left,right]})
pairs.sort_values('abs_correlation',ascending=False).to_csv(REPORT/'high_correlation_pairs.csv',index=False)
print('Correlation sample size:',len(sample_idx),'pairs with |r| >= 0.98:',len(pairs))
display(pairs.sort_values('abs_correlation',ascending=False).head(15))
fig,axes=plt.subplots(1,3,figsize=(15,4))
axes[0].hist(X[sample_idx,0],bins=60); axes[0].set(title='Raw file size',xlabel='Bytes')
axes[1].hist(np.log1p(X[sample_idx,0]),bins=60); axes[1].set(title='log1p file size',xlabel='log1p bytes')
axes[2].hist(Z[:,np.flatnonzero(variable==0)[0]],bins=60); axes[2].set(title='Standardized file size',xlabel='Fitting z-score')
savefig('05_normalization_and_tails.png')
print('PASS: normalization derives only from fitting statistics; constant filtering and redundancy audit complete.')
''','7_normalization_redundancy')
md('''## 8. Training-only feature selection: ANOVA and mutual information

Compute exact binary ANOVA F scores from all fitting rows in bounded blocks. For category scores, use only annotated malicious fitting records; other observed behaviors are weak proxy negatives. A category with insufficient fitting support is skipped. P-values are descriptive multiple-comparison quantities, not evidence of causal or operational usefulness. No classifier is trained to obtain these scores.

Estimate mutual information only for maliciousness on a deterministic 6,000-row fitting sample; MI is slower and sample-sensitive. Mark the PE flag/start bytes as discrete. Other inputs are treated as continuous; sparse/zero-heavy features can make the estimate unstable. Aggregate within-target percentile ranks equally across supported categories plus maliciousness so a common category does not dominate through raw F-score magnitude. Export top-64/128/256 **candidate** subsets and a correlation-pruned candidate set. These are not validated best features; that decision belongs to a later supervised milestone.''')
code('''roles={'maliciousness':(np.ones(len(idx['fit']),dtype=bool),y[idx['fit']])}
for j,label in enumerate(LABELS):
    if supported[j]: roles[label]=(((y[idx['fit']]==1)&annotated[idx['fit']]),Y[idx['fit'],j])
anova={role:np.full(D,np.nan) for role in roles}; pvalues={role:np.full(D,np.nan) for role in roles}
for start in range(0,D,16):
    columns=np.arange(start,min(start+16,D)); a=np.asarray(X[np.ix_(idx['fit'],columns)],dtype=np.float64)
    a=np.where(np.isfinite(a),a,medians[columns])
    for role,(mask,target) in roles.items():
        f,p=binary_anova(a[mask],target[mask]); anova[role][columns]=f; pvalues[role][columns]=p
    del a
ranking=dictionary.copy()
for role in roles:
    ranking['F_'+role]=anova[role]; ranking['p_'+role]=pvalues[role]
percentile=pd.DataFrame({role:pd.Series(v).rank(pct=True).fillna(0) for role,v in anova.items()})
ranking['consensus_rank']=percentile.mean(axis=1)
from sklearn.feature_selection import mutual_info_classif
mi_idx=np.array(hash_order[:6000]); mi_input=np.asarray(X[np.ix_(mi_idx,variable)],dtype=np.float64)
mi_input=np.where(np.isfinite(mi_input),mi_input,medians[variable])
discrete=np.isin(variable,[2,3,4,5,6])
ranking['MI_maliciousness']=np.nan
ranking.loc[variable,'MI_maliciousness']=mutual_info_classif(mi_input,y[mi_idx],discrete_features=discrete,random_state=SEED)
del mi_input
ranking['constant']=stats.constant.to_numpy()
ranking=ranking.sort_values('consensus_rank',ascending=False)
ranking.to_csv(REPORT/'feature_selection_ranking.csv',index=False)
eligible=ranking.loc[~ranking.constant,'index'].to_list()
position={int(j):k for k,j in enumerate(variable)}; pruned=[]
for j in eligible:
    if not pruned or np.max(np.abs(correlation[position[j],[position[k] for k in pruned]]))<.98: pruned.append(j)
candidates={f'top_{k}':eligible[:k] for k in [64,128,256]}
candidates['correlation_pruned']=pruned
selection={'status':'exploratory filter subsets; not classifier-validated','fitting_records':len(idx['fit']),
 'category_records':int(mal_annotated[meta.row_id.iloc[idx['fit']]].sum()),'MI_sample_n':len(mi_idx),
 'targets':list(roles),'feature_cache_key':key,'candidates':{name:[names[j] for j in ids] for name,ids in candidates.items()}}
(REPORT/'candidate_feature_subsets.json').write_text(json.dumps(selection,indent=2))
display(ranking[['index','feature','group','consensus_rank','MI_maliciousness']].head(20))
fig,axes=plt.subplots(1,2,figsize=(16,6))
ranking[~ranking.constant].head(18).sort_values('consensus_rank').set_index('feature').consensus_rank.plot.barh(ax=axes[0],title='Equal-target ANOVA rank consensus')
mi_top=ranking.dropna(subset=['MI_maliciousness']).nlargest(18,'MI_maliciousness').sort_values('MI_maliciousness')
mi_top.set_index('feature').MI_maliciousness.plot.barh(ax=axes[1],title='Maliciousness MI (6,000 fitting rows)')
savefig('06_feature_rankings.png')
heat_features=eligible[:35]; positions=[position[j] for j in heat_features]
fig,ax=plt.subplots(figsize=(11,9)); im=ax.imshow(correlation[np.ix_(positions,positions)],vmin=-1,vmax=1,cmap='coolwarm')
ax.set_xticks(range(len(positions)),[names[j] for j in heat_features],rotation=90,fontsize=7)
ax.set_yticks(range(len(positions)),[names[j] for j in heat_features],fontsize=7); ax.set_title('Correlation among top-ranked fitting features'); fig.colorbar(im,ax=ax)
savefig('07_ranked_feature_correlation.png')
assert all(not stats.constant.iloc[j] for ids in candidates.values() for j in ids)
print('Correlation-pruned candidate dimensions:',len(pruned))
print('PASS: fitting-only univariate rankings and candidate feature subsets exported; no classifier fit or score performed.')
''','8_feature_selection')
md('''## 9. PCA: variance structure and visual separation

Fit PCA on the normalized fitting sample only. Its complete eigenvalue profile shows how many components explain 90/95/99% of sampled standardized variance. Project those same fitting points into two dimensions and color by format or label for interpretation. Labels never enter PCA fitting. PCA may primarily encode file format or nuisance variation; variance retention and visually separated clusters are not classification accuracy.

Export variance and top-loading tables, not a serialized fitted model. Sampling and standardization choices are recorded; low-dimensional views cannot establish that a feature subset will generalize.''')
code('''from sklearn.decomposition import PCA
pca=PCA(svd_solver='covariance_eigh').fit(Z)
explained=pd.DataFrame({'component':np.arange(1,len(pca.explained_variance_ratio_)+1),'variance_ratio':pca.explained_variance_ratio_})
explained['cumulative_variance']=explained.variance_ratio.cumsum(); explained.to_csv(REPORT/'pca_explained_variance.csv',index=False)
components_needed={str(q):int(np.searchsorted(explained.cumulative_variance.to_numpy(),q)+1) for q in [.9,.95,.99]}
coordinates=(Z[:6000]-pca.mean_)@pca.components_[:2].T
loadings=pd.DataFrame({'feature':[names[j] for j in variable],'PC1':pca.components_[0],'PC2':pca.components_[1]})
loadings.to_csv(REPORT/'pca_loadings.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(17,5))
axes[0].plot(explained.component,explained.cumulative_variance); axes[0].axhline(.95,color='gray',ls='--'); axes[0].set(xlabel='Components',ylabel='Cumulative variance',title='Standardized fitting variance')
color_meta=meta.iloc[sample_idx[:6000]]
for ft in sorted(color_meta.file_type.unique()):
    take=color_meta.file_type.eq(ft).to_numpy(); axes[1].scatter(coordinates[take,0],coordinates[take,1],s=5,alpha=.3,label=ft,rasterized=True)
axes[1].legend(); axes[1].set(title='Fitting PCA by format',xlabel='PC1',ylabel='PC2')
for value,label in [(0,'Benign'),(1,'Malicious')]:
    take=y[sample_idx[:6000]]==value; axes[2].scatter(coordinates[take,0],coordinates[take,1],s=5,alpha=.3,label=label,rasterized=True)
axes[2].legend(); axes[2].set(title='Same projection, label overlay',xlabel='PC1',ylabel='PC2')
savefig('08_pca_structure.png')
print('Sampled components needed:',components_needed)
assert np.isfinite(coordinates).all() and np.isclose(explained.variance_ratio.sum(),1)
print('PASS: PCA fitted only on fitting inputs; variance and loading evidence exported without a classifier.')
''','9_pca_visualization')
md('''## 10. Fitting-to-validation feature drift

Compare deterministic, label-independent feature samples from fitting and validation periods. Use KS distance as a distributional effect size, not a p-value claim about independent observations; related files/time dependence violate simple IID interpretations. Rank features by KS distance and visualize the largest changes. Validation labels do not influence feature selection or PCA. Test/challenge inputs are not used for these drift rankings.

Drift is descriptive evidence for the next design discussion. It is not a reason to tune repeatedly against held-out labels or silently replace the official evaluation population.''')
code('''from scipy.stats import ks_2samp
val_order=sorted(idx['validation'],key=lambda i:hashlib.sha256((str(SEED)+meta.sha256.iloc[i]).encode()).digest())
val_idx=np.array(val_order[:20000]); drift_rows=[]
for start in range(0,D,16):
    columns=np.arange(start,min(start+16,D)); a=np.asarray(X[np.ix_(sample_idx,columns)]); b=np.asarray(X[np.ix_(val_idx,columns)])
    for k,j in enumerate(columns):
        aa=a[:,k]; bb=b[:,k]; aa=aa[np.isfinite(aa)]; bb=bb[np.isfinite(bb)]
        distance=ks_2samp(aa,bb,method='asymp').statistic if len(aa) and len(bb) else np.nan
        drift_rows.append({'index':int(j),'feature':names[j],'group':names[j].split('.')[0],'ks_distance':float(distance),
            'fit_median':float(np.median(aa)),'validation_median':float(np.median(bb))})
drift=pd.DataFrame(drift_rows).sort_values('ks_distance',ascending=False)
drift.to_csv(REPORT/'fitting_validation_drift.csv',index=False); display(drift.head(15))
fig,axes=plt.subplots(1,2,figsize=(15,6))
drift.head(18).sort_values('ks_distance').set_index('feature').ks_distance.plot.barh(ax=axes[0],title='Largest fitting → validation KS distances')
for split,ii in [('Fitting',sample_idx),('Validation',val_idx)]:
    axes[1].hist(np.log1p(X[ii,0]),bins=60,density=True,histtype='step',lw=1.5,label=split)
axes[1].legend(); axes[1].set(title='Size distribution over time',xlabel='log1p bytes',ylabel='Density')
savefig('09_temporal_feature_drift.png')
assert set(meta.iloc[sample_idx].split)=={'fit'} and set(meta.iloc[val_idx].split)=={'validation'}
print('PASS: drift inspection uses fitting/validation inputs only, with no held-out feature ranking.')
''','10_distribution_drift')
md('''## 11. Preserved baseline scores and this milestone's conclusions

The following scores were produced by the earlier completed experiments. Their classifier files have been removed at the user's request. Read the protected CSVs as historical reference only: no predictive model is loaded, retrained, or scored by this notebook.

Stop this milestone with documented candidate inputs, data-quality concerns and questions for the next discussion. Univariate associations, PCA variance and drift do not establish the best classifier or pipeline.''')
code('''verify_score_reference(P.historical)
baseline=pd.read_csv(P.historical/'ember2024_baseline_comparison.csv')
columns=['split','configuration','n','detection_recall','detection_fpr','behavior_micro_f1','behavior_macro_ap']
display(baseline[columns])
unsupported=[label for label,ok in zip(LABELS,supported) if not ok]
summary={'milestone':'primary data and feature analysis only','classifiers_trained':0,'serialized_models':0,
 'raw_rows':quality['raw_rows'],'retained_metadata_rows':len(meta),'eligible_rows':int(keep.sum()),
 'fitting_rows':len(idx['fit']),'fitting_formats':sorted(meta.iloc[idx['fit']].file_type.unique().tolist()),
 'features':D,'variable_features':len(variable),'constant_features':int(stats.constant.sum()),
 'supported_roles':[label for label,ok in zip(LABELS,supported) if ok],'unsupported_roles':unsupported,
 'unannotated_fitting_malware':int((fit&(y==1)&~annotated).sum()),
 'duplicate_audit':conflict_summary,'tlsH_cross_partition_purges':tlsh_purged,'vector_quality':vector_quality,
 'correlation_pairs_098':len(pairs),'correlation_pruned_dimensions':len(pruned),
 'correlation_and_PCA_fit_sample_n':len(sample_idx),'PCA_components_needed':components_needed,
 'source_records_sha256':source_sha,'historical_score_status':score_reference['status']}
(REPORT/'analysis_summary.json').write_text(json.dumps(summary,indent=2))
notes=f"""# Primary analysis findings

This milestone ingested {quality['raw_rows']:,} raw records and retained {int(keep.sum()):,} eligible records after identity, TLSH and feature-vector controls. The fitting period contains {len(idx['fit']):,} records across {', '.join(summary['fitting_formats'])}.

The shared static representation has {D} named inputs: {len(variable)} variable and {int(stats.constant.sum())} constant fitting columns. There are {nonfinite:,} non-finite vector cells before fitting-median handling. Independent raw-to-vector checks passed for {checked} rows spanning all file types.

Fitting malware without behavior annotations: {summary['unannotated_fitting_malware']:,}. Unsupported mapped roles: {', '.join(unsupported) if unsupported else 'none at the stated support threshold'}. Duplicate identities with binary-label disagreement: {conflict_summary['binary_conflict_identities']:,}; with behavior-tag disagreement: {conflict_summary['behavior_conflict_identities']:,}. Review these audits before trusting labels as complete ground truth.

ANOVA uses full fitting-only data; mutual information uses {len(mi_idx):,} deterministic fitting records. Candidate top-64/128/256 and correlation-pruned subsets are proposals, not validated winners. There are {len(pairs):,} fitting-sample feature pairs with absolute Pearson correlation at least 0.98. PCA on {len(sample_idx):,} fitting records needs {components_needed['0.95']} components for 95% of sampled standardized variance. PCA does not measure predictive accuracy.

The strongest univariate features and temporal distribution changes are in the ranking/drift tables and figures. Their associations may reflect format, family, collection or label biases. Do not infer causal behavior from a ranked feature or treat a visually separated projection as a classifier result.

The historical baseline comparison is retained with a checksum under `results/historical/`. Its earlier 600,000-record APK/PDF/Win64 fitting population differs from the current 400,000-record Win64 fitting population. Other older supporting reports are no longer present; recovery provenance is recorded in its manifest. All saved classifier artifacts and prior selected thresholds were removed. No classifier was trained or evaluated in this milestone.

## Questions to review before the next step

1. Should we retain only shared inputs or analyze PE-specific features separately for Win64?
2. Which category aliases and missing-annotation assumptions are defensible, especially unsupported Botnet Agent?
3. Which controlled feature subsets and transforms merit comparison in a later fitting-only supervised experiment?
4. Should similarity controls extend beyond exact matches to fuzzy/family grouping?
5. What new untouched evaluation period can support future claims, given the existing historical holdouts have already been inspected?
"""
(REPORT/'findings.md').write_text(notes); display(Markdown(notes))
assert_no_models(ART); verify_score_reference(P.historical)
print('PASS: primary analysis complete; zero classifier fits, zero saved models, baseline reference verified.')
''','11_findings_reference')
from multi_label_static_analysis.notebook_readability import readable_notebook
nb.cells=cells
nb=readable_notebook(nb,kind='analysis'); out.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,out); print(out)
