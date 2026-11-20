"""Notebook source fragment: exact fitting statistics in bounded column blocks."""
feature_names=[f'{group}_{i}' for group,columns in groups.items() for i in range(len(columns))]
stat_records=[]
# Calculate exact statistics on all fitting rows without materializing the full matrix.
for start in range(0,D,16):
    columns=np.arange(start,min(start+16,D))
    values=np.asarray(X[np.ix_(idx['fit'],columns)])
    summary={'missing_rate':np.isnan(values).mean(axis=0),'zero_rate':(values==0).mean(axis=0),
       'mean':np.nanmean(values,axis=0,dtype=np.float64),'std':np.nanstd(values,axis=0,dtype=np.float64),
       'p01':np.nanpercentile(values,1,axis=0),'median':np.nanmedian(values,axis=0),
       'p99':np.nanpercentile(values,99,axis=0)}
    for k,j in enumerate(columns):
        stat_records.append({'feature':feature_names[j],**{name:float(v[k]) for name,v in summary.items()}})
    del values
feature_stats=pd.DataFrame(stat_records)
feature_stats['constant']=feature_stats['std'].eq(0)
feature_stats.to_csv(REPORT/'feature_statistics.csv',index=False)
display(feature_stats.groupby(feature_stats.feature.str.replace(r'_\d+$','',regex=True)).agg(
    dimensions=('feature','size'),constant_columns=('constant','sum'),mean_zero_fraction=('zero_rate','mean')))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ft in sorted(meta.iloc[idx['fit']].file_type.unique()):
    take=idx['fit'][meta.iloc[idx['fit']].file_type.eq(ft).to_numpy()]
    axes[0].hist(np.log10(1+X[take,0]),bins=60,alpha=.4,density=True,label=ft)
    axes[1].hist(X[take,1],bins=60,alpha=.4,density=True,label=ft)
axes[0].set(xlabel='log10(1 + size in bytes)',ylabel='Density',title='Fitting file size distribution')
axes[1].set(xlabel='Byte entropy',ylabel='Density',title='Fitting entropy distribution')
for ax in axes: ax.legend()
plt.tight_layout(); fig.savefig(REPORT/'figures/03_feature_distributions.png',dpi=150); plt.show()
assert len(feature_stats)==D
print('PASS: exact feature statistics computed on all fitting rows in bounded column blocks.')
