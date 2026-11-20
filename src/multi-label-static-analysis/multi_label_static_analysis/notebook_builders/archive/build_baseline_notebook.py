"""Build the reproducible published-configuration EMBER2024 comparison notebook."""
from pathlib import Path
import argparse
import nbformat as nbf
from multi_label_static_analysis.config import project_paths

PROJECT=project_paths()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=PROJECT.baseline_notebook)
parser.add_argument('--force',action='store_true')
args=parser.parse_args(); out=args.output.resolve()
if out.exists() and not args.force: parser.error(f'{out} exists; use --force to replace it.')
main=nbf.read(PROJECT.notebook,4)
nb=nbf.v4.new_notebook(metadata=dict(main.metadata)); cells=[]
def md(source): cells.append(nbf.v4.new_markdown_cell(source))
def code(source,stage):
 c=nbf.v4.new_code_cell(source); c.metadata['stage']=stage; cells.append(c)
md('''# Official EMBER2024 configuration: local multi-label baseline comparison

This companion notebook retrains the **published one-vs-rest LightGBM configuration** on the same APK/PDF/Win64 fitting period, shared 696 static inputs, supported proposal categories, chronological splits, and observed-target evaluation masks as the research notebook. It compares the resulting direct category predictions and a gated variant with the frozen selected research pipeline.

**Official sources:** [tag configuration](https://github.com/FutureComputing4AI/EMBER2024/blob/0ef753e81d98bf209f71b03cd331dfc190b5b54d/examples/lgbm_config_tag.json), [detection configuration](https://github.com/FutureComputing4AI/EMBER2024/blob/0ef753e81d98bf209f71b03cd331dfc190b5b54d/examples/lgbm_config.json), [OvR implementation](https://github.com/FutureComputing4AI/EMBER2024/blob/0ef753e81d98bf209f71b03cd331dfc190b5b54d/src/thrember/model.py), and [paper §4.6](https://arxiv.org/abs/2506.05074). Checked-in JSON files and their SHA256 hashes pin the reference configuration; a normal run does not need network access.

The published behavior model expects **2,568 features**, whereas the available APK/PDF/Win64 study uses **696**. We do not pass incompatible vectors to that pretrained model, and do not describe this local refit as an exact reproduction of the full-data published benchmark.

**Explicit adaptations:** common non-PE inputs; training-supported mapped categories instead of the paper's 92 behavior tags; chronological validation instead of the library's random 10% split; exclude unannotated malicious targets instead of assuming negatives; categorical indices restricted to valid shared columns 2–6 (PE columns 701–702 do not exist); four CPU threads and quieter logs. Model seed 0 and all statistical hyperparameters come from the official JSON. The detector uses 500 rounds/64 leaves/minimum 100 samples per leaf; tag heads use up to 100 rounds/64 leaves/minimum 10, with 10-round validation early stopping. Selected tag iteration counts are then refit on the full fitting period.

The direct OvR model trains on benign plus annotated-malicious records. A gated variant combines those heads with the official-configuration detector. The research category heads instead train conditionally on annotated malware; this training-population distinction is reported, rather than attributed solely to hyperparameters. Unsupported categories remain unpredicted.

This comparison was added after the original study's test results were available. It uses the same frozen evaluation protocol and no test-driven tuning; interpret it as an additional descriptive benchmark, not a new blind holdout study.''')
for stage in ['1_data_audit','2_statistics','3_vectorization']:
 original=next(c for c in main.cells if c.cell_type=='code' and c.metadata.get('stage')==stage)
 md('## '+stage.replace('_',' ',1).replace('_',' '))
 code(original.source,stage)
md('''## 4. Fit the published configuration

Use native LightGBM and preserve all 696 original column positions. Tree models use native missing-value handling and no normalization/PCA. General categorical columns retain their fixed indices. Each tag classifier is a binary one-vs-rest model with the official imbalance and sampling settings. Early stopping is selected using the configuration-validation window; calibration/test/challenge targets are not passed to fitting. All baseline fits use the full eligible fitting period, rather than the research candidate-search sample.''')
code('''import joblib
import lightgbm as lgb
from sklearn.metrics import roc_curve, multilabel_confusion_matrix
CONFIG_DIR=ROOT/'configs/ember2024'
provenance=json.loads((CONFIG_DIR/'provenance.json').read_text())
for name,info in provenance['files'].items():
    assert hashlib.sha256((CONFIG_DIR/name).read_bytes()).hexdigest()==info['sha256']
tag_config=json.loads((CONFIG_DIR/'lgbm_config_tag.json').read_text())
gate_config=json.loads((CONFIG_DIR/'lgbm_config.json').read_text())
assert tag_config['num_leaves']==64 and tag_config['num_iterations']==100
assert gate_config['num_iterations']==500
active_ids=np.flatnonzero(active)
fi=idx['fit']; vi=idx['validation']; ci=idx['calibration']
observed_fit=fi[behavior_observed[fi]]; observed_val=vi[behavior_observed[vi]]
categorical=[2,3,4,5,6]
fit_records=[]
def native_fit(config,ii,target,role,validation=None,rounds=None):
    params=dict(config)
    configured_rounds=int(params.pop('num_iterations'))
    patience=params.pop('early_stopping_rounds',None)
    params.update(num_threads=4,verbosity=-1)
    dataset=lgb.Dataset(X[ii],label=target,categorical_feature=categorical,free_raw_data=True)
    kw={}
    if validation is not None:
        valid_ii,valid_y=validation
        kw['valid_sets']=[lgb.Dataset(X[valid_ii],label=valid_y,reference=dataset,categorical_feature=categorical)]
        kw['callbacks']=[lgb.early_stopping(int(patience),first_metric_only=True,verbose=False)]
    start=time.perf_counter()
    model=lgb.train(params,dataset,num_boost_round=rounds or configured_rounds,**kw)
    fit_records.append({'role':role,'rows':len(ii),'rounds':model.current_iteration(),
                        'best_iteration':model.best_iteration,'seconds':time.perf_counter()-start})
    return model
baseline_gate=native_fit(gate_config,fi,y[fi],'official_detector')
print('Fitted official-configuration detector',flush=True)
baseline_heads={}; selected_rounds={}
for j in active_ids:
    assert len(np.unique(Y[observed_fit,j]))==2
    tuned=native_fit(tag_config,observed_fit,Y[observed_fit,j],LABELS[j]+'_validation',
                     validation=(observed_val,Y[observed_val,j]))
    rounds=tuned.best_iteration or tuned.current_iteration()
    selected_rounds[LABELS[j]]=int(rounds)
    baseline_heads[j]=native_fit(tag_config,observed_fit,Y[observed_fit,j],LABELS[j]+'_refit',rounds=rounds)
    print('Fitted',LABELS[j],':',rounds,'boosting rounds',flush=True)
fit_table=pd.DataFrame(fit_records); fit_table.to_csv(REPORT/'ember2024_baseline_training.csv',index=False)
display(fit_table)
research=joblib.load(ART/'selected_models.joblib')
assert 'two_stage' in research['thresholds'], 'Missing frozen research two-stage comparator.'
assert np.array_equal(research['active'],active)
assert research['labels']==LABELS
assert all(np.isfinite(m.predict(X[ci[:5]],num_threads=4)).all() for m in baseline_heads.values())
print('PASS: pinned official configuration verified; detector and supported OvR heads trained without test/challenge data.')
''','4_baseline_training')
md('''## 5. Freeze comparable operating points

Use the existing later calibration window. Both detectors choose the highest recall with empirical false-positive rate ≤1%. Per-label thresholds maximize calibration end-to-end F1 on observed targets. Direct OvR predictions do not receive a gate veto. The gated baseline multiplies detector and category scores and vetoes labels below the detector threshold, matching the research pipeline's decision rule. The existing research thresholds are loaded unchanged. Scores are uncalibrated; probability products are ranking scores.

For the direct variant the detection metrics describe the independent official detector; its category outputs remain ungated. Report the rate of any false category assignment to benign records separately to expose this distinction.''')
code('''def ap_safe(t,p):
    return float(average_precision_score(t,p)) if len(np.unique(t))==2 else np.nan

def choose_gate(t,p,budget=.01):
    fpr,tpr,threshold=roc_curve(t,p,drop_intermediate=False)
    feasible=np.flatnonzero(fpr<=budget)
    best=feasible[np.lexsort((-threshold[feasible],fpr[feasible],-tpr[feasible]))[0]]
    return float(threshold[best])

def choose_label(t,p,gate):
    if len(np.unique(t))<2: return .5
    precision,recall,threshold=precision_recall_curve(t,np.where(gate,p,-1.))
    f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    valid=np.flatnonzero(threshold>=0)
    return float(threshold[valid[np.argmax(f[valid])]]) if len(valid) else 1.

def baseline_scores(ii):
    gate=baseline_gate.predict(X[ii],num_threads=4)
    head=np.zeros((len(ii),7))
    for j,model in baseline_heads.items(): head[:,j]=model.predict(X[ii],num_threads=4)
    return gate,head

g,h=baseline_scores(ci); gt=choose_gate(y[ci],g); observed=behavior_observed[ci]
operating={}
for mode in ['ember2024_ovr_direct','ember2024_ovr_gated']:
    gated=mode.endswith('_gated')
    p=g[:,None]*h if gated else h
    veto=g>=gt if gated else np.ones(len(ci),dtype=bool)
    ht=np.full(7,np.inf)
    for j in active_ids: ht[j]=choose_label(Y[ci][observed,j],p[observed,j],veto[observed])
    operating[mode]={'gate':gt,'behavior':ht}
calibration_fpr=float((g[y[ci]==0]>=gt).mean())
assert calibration_fpr<=.01+1e-12
print('Official-configuration calibration detector FPR:',calibration_fpr)
print('Baseline gate threshold:',gt)
selection={'source_revision':provenance['revision'],'feature_count':D,'categorical_columns':categorical,
    'fitting_rows':len(fi),'category_training_rows':len(observed_fit),'supported_labels':[LABELS[j] for j in active_ids],
    'selected_tag_rounds':selected_rounds,'calibration_fpr':calibration_fpr,
    'source_records_sha256':fingerprint.hexdigest(),'feature_cache_key':key,
    'adaptations':['696 shared inputs instead of pretrained 2568','chronological validation','supported mapped categories instead of 92 tags',
       'exclude missing malware annotations','categorical indices 2-6 only','four CPU threads; quieter logs'],
    'thresholds':{k:{'gate':v['gate'],'behavior':[float(t) if np.isfinite(t) else None for t in v['behavior']]} for k,v in operating.items()}}
(ART/'ember2024_baseline_selection.json').write_text(json.dumps(selection,indent=2))
joblib.dump({'gate':baseline_gate,'heads':baseline_heads,'operating':operating,'active':active,'labels':LABELS,
    'selection':selection,'official_tag_config':tag_config,'official_gate_config':gate_config},ART/'ember2024_baseline_models.joblib')
print('PASS: baseline thresholds frozen on calibration only; official-config models and provenance saved.')
''','5_baseline_calibration')
md('''## 6. Held-out comparison and interpretation

Score identical test/challenge records for all configurations. Category metrics use the identical observed-target mask and supported-category set. Include direct and gated OvR, per-category support, per-format detection, a 200-replicate bootstrap of paired week clusters for behavior micro-F1 differences, and gate-error effects. The reference pipeline is frozen; no test-driven model selection occurs here.

The original paper's 92-tag behavior macro-F1 (0.1345) is context only. It is not directly comparable to these mapped-category, APK/PDF/Win64-only, incomplete-annotation results. Malicious-only challenge detection AP, ROC AUC, and FPR remain undefined.''')
code('''def behavior_metrics(t,p,pred):
    t=t[:,active_ids]; p=p[:,active_ids]; pred=pred[:,active_ids]
    valid=[k for k in range(t.shape[1]) if len(np.unique(t[:,k]))==2]
    aps=[ap_safe(t[:,j],p[:,j]) for j in valid]
    return {'behavior_micro_f1':f1_score(t,pred,average='micro',zero_division=0),
       'behavior_macro_f1':f1_score(t[:,valid],pred[:,valid],average='macro',zero_division=0) if valid else np.nan,
       'behavior_macro_ap':float(np.mean(aps)) if aps else np.nan,
       'behavior_micro_ap':ap_safe(t.ravel(),p.ravel()),
       'behavior_macro_precision':precision_score(t[:,valid],pred[:,valid],average='macro',zero_division=0) if valid else np.nan,
       'behavior_macro_recall':recall_score(t[:,valid],pred[:,valid],average='macro',zero_division=0) if valid else np.nan,
       'subset_accuracy':accuracy_score(t,pred),'hamming_loss':hamming_loss(t,pred)}

def gate_metrics(t,p,pred):
    tn,fp,fn,tp=confusion_matrix(t,pred,labels=[0,1]).ravel()
    return {'detection_precision':precision_score(t,pred,zero_division=0),
       'detection_recall':recall_score(t,pred,zero_division=0),'detection_f1':f1_score(t,pred,zero_division=0),
       'detection_ap':ap_safe(t,p),'detection_roc_auc':roc_auc_score(t,p) if len(np.unique(t))==2 else np.nan,
       'detection_fpr':fp/(fp+tn) if fp+tn else np.nan,'false_alarms':int(fp),'missed_malware':int(fn)}

rows=[]; category_rows=[]; format_rows=[]; bootstrap_rows=[]; all_predictions={}
for split in ['test','challenge']:
    ii=idx[split]; observed=behavior_observed[ii]
    bg,bh=baseline_scores(ii)
    rg=research['gate'].predict_proba(X[ii])[:,1]; rh=np.zeros((len(ii),7))
    for j,model in research['heads'].items(): rh[:,j]=model.predict_proba(X[ii])[:,1]
    score_sets={'research_two_stage':(rg,rg[:,None]*rh,research['thresholds']['two_stage'],True),
       'ember2024_ovr_direct':(bg,bh,operating['ember2024_ovr_direct'],False),
       'ember2024_ovr_gated':(bg,bg[:,None]*bh,operating['ember2024_ovr_gated'],True)}
    for name,(g,p,op,gated) in score_sets.items():
        gate=g>=op['gate']; pred=p>=op['behavior']
        if gated: pred &= gate[:,None]
        pred[:,~active]=False
        benign=y[ii]==0
        row={'split':split,'configuration':name,'n':len(ii),'behavior_n':int(observed.sum()),
          'behavior_coverage':float(observed.mean()),**gate_metrics(y[ii],g,gate),
          **behavior_metrics(Y[ii][observed],p[observed],pred[observed]),
          'benign_any_category_fpr':float(pred[benign][:,active_ids].any(axis=1).mean()) if benign.any() else np.nan}
        rows.append(row); all_predictions[(split,name)]=(ii,p,pred)
        for j,label in enumerate(LABELS):
            t=Y[ii][observed,j]; pr=pred[observed,j]
            category_rows.append({'split':split,'configuration':name,'category':label,'estimable':bool(active[j]),
              'positives':int(t.sum()),'f1':f1_score(t,pr,zero_division=0) if active[j] and t.sum() else np.nan,
              'precision':precision_score(t,pr,zero_division=0) if active[j] else np.nan,
              'recall':recall_score(t,pr,zero_division=0) if active[j] and t.sum() else np.nan,
              'ap':ap_safe(t,p[observed,j]) if active[j] else np.nan})
        for ft in meta.iloc[ii].file_type.unique():
            mask=meta.iloc[ii].file_type.eq(ft).to_numpy()
            format_rows.append({'split':split,'configuration':name,'file_type':ft,'n':int(mask.sum()),
                                **gate_metrics(y[ii][mask],g[mask],gate[mask])})
        print('Scored',split,name,flush=True)
    weeks=meta.iloc[ii].date.dt.strftime('%G-%V').to_numpy(); unique=np.unique(weeks)
    members=[np.flatnonzero(weeks==w) for w in unique]
    # Aggregate sufficient TP/FP/FN counts by week for efficient paired resampling.
    counts={}
    for name in score_sets:
        pred=all_predictions[(split,name)][2]; c=[]
        for take in members:
            take=take[observed[take]]; t=Y[ii][take][:,active_ids].astype(bool); pr=pred[take][:,active_ids]
            c.append([np.sum(t&pr),np.sum(~t&pr),np.sum(t&~pr)])
        counts[name]=np.asarray(c)
    rng=np.random.default_rng(SEED)
    differences={name:[] for name in score_sets if name!='research_two_stage'}
    for _ in range(200):
        take=rng.integers(0,len(unique),len(unique)); f={}
        for name,c in counts.items():
            tp,fp,fn=c[take].sum(axis=0); f[name]=2*tp/max(1,2*tp+fp+fn)
        for name in differences: differences[name].append(f[name]-f['research_two_stage'])
    for name,v in differences.items():
        bootstrap_rows.append({'split':split,'baseline':name,'difference':'baseline minus research behavior micro-F1',
            'weeks':len(unique),'replicates':200,'lower95':float(np.quantile(v,.025)),'upper95':float(np.quantile(v,.975))})
comparison=pd.DataFrame(rows); per_category=pd.DataFrame(category_rows); by_format=pd.DataFrame(format_rows)
comparison.to_csv(REPORT/'ember2024_baseline_comparison.csv',index=False)
per_category.to_csv(REPORT/'ember2024_baseline_categories.csv',index=False)
by_format.to_csv(REPORT/'ember2024_baseline_filetypes.csv',index=False)
pd.DataFrame(bootstrap_rows).to_csv(REPORT/'ember2024_baseline_paired_intervals.csv',index=False)
# Independently reproduce the prior pipeline scores as an integration check.
prior=pd.read_csv(REPORT/'final_metrics.csv')
for split in ['test','challenge']:
    old=prior.query('split == @split and architecture == "two_stage"').iloc[0]
    new=comparison.query('split == @split and configuration == "research_two_stage"').iloc[0]
    assert np.isclose(old.recall,new.detection_recall)
    assert np.isclose(old.behavior_micro_f1,new.behavior_micro_f1)
assert comparison.query('split == "challenge"').detection_fpr.isna().all()
display(comparison); display(per_category); display(pd.DataFrame(bootstrap_rows))
fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,split in zip(axes,['test','challenge']):
    part=comparison.query('split == @split').set_index('configuration')
    part[['detection_recall','behavior_micro_f1']].plot.bar(ax=ax,rot=15)
    ax.set(title=split.capitalize()+' frozen comparison',ylabel='Score',ylim=(0,1))
plt.tight_layout(); fig.savefig(REPORT/'figures/04_ember2024_baseline.png',dpi=150,bbox_inches='tight'); plt.show()
report="""# EMBER2024 published-configuration comparison

The official one-vs-rest LightGBM configuration was refit on the local chronological APK/PDF/Win64 study. The pretrained behavior model expects 2,568 inputs and was not used with the incompatible 696-input representation. This is a local configuration baseline, not a reproduction of the paper's full-data/92-tag scores.

The direct variant leaves category outputs ungated; its detection metrics describe an independent official-configuration detector. The gated variant applies the detector veto and probability-score product used by the research pipeline. All configurations use the same held-out records and observed-target evaluation mask. Native baseline inputs retain all 696 columns; the research model filters constants. Thresholds are fixed using the same calibration period.

## Results

"""+comparison[['split','configuration','detection_recall','detection_fpr','detection_ap','behavior_micro_f1','behavior_macro_ap','benign_any_category_fpr']].to_csv(index=False)+"""

## Interpretation limits

Configurations use the fitting-supported mapped categories. Targets without fitting support remain untrained. The paper's 92-tag macro-F1 is not directly comparable. Native baseline heads train on observed benign plus annotated-malicious rows, whereas research heads train on annotated malware. Chronological validation replaces the official helper's random split; nonexistent PE categorical indices are omitted. Tag rounds are selected with early stopping on validation, then refit on the entire fitting period. Four CPU threads and quieter logs are execution-only adaptations. Original statistical hyperparameters and seed 0 are recorded in the pinned JSON configs and baseline model bundle.

The baseline was added after original test results were available; this is a descriptive comparison without test-driven tuning. Weak annotations, rare categories, single-seed training, and challenge format shift remain. Paired week-bootstrap intervals quantify observed week variability in micro-F1 differences, not general superiority over future data.

See ember2024_baseline_categories.csv, ember2024_baseline_filetypes.csv, ember2024_baseline_paired_intervals.csv, and ember2024_baseline_training.csv for details. Source revision, checksums, categorical indices, training sizes, selected rounds, and frozen thresholds are recorded in configs/ember2024/provenance.json and artifacts/ember2024_baseline_selection.json.
"""
(REPORT/'ember2024_baseline_findings.md').write_text(report)
display(Markdown(report))
print('PASS: baseline comparison scored on identical held-out records; frozen research metrics reproduced; outputs exported.')
''','6_baseline_evaluation')
nb.cells=cells; out.parent.mkdir(parents=True,exist_ok=True); nbf.write(nb,out); print(out)
