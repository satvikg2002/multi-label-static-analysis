"""Consistent, small teaching cells for the two current notebooks.

The builders remain the source of the analysis. This layout layer changes cell
boundaries, formatting, comments and explanatory prose, not the calculations.
"""
import ast
import textwrap
import nbformat

# Each boundary begins one task. Indices refer to top-level statements in the
# original section, before formatting or extracting a contained plotting helper.
PLANS = {
    '1_scope_inventory': [(0,'Import analysis tools','Limit numerical threads so the notebook stays responsive.'),(13,'Choose paths and settings','Use the TOML paths; keep the seed and display settings together.'),(22,'Define output helpers','Each plot is saved before display; each table is exported before display.'),(24,'Check inputs and historical references','Fail early if classifier artifacts remain or preserved scores changed.')],
    '2_ingestion': [(0,'Ingest metadata once','Stream raw JSONL rather than retaining complete feature records in RAM.'),(3,'Normalize metadata types','Unparseable values remain visible as missing values for the audit.'),(7,'Export the source inventory','Keep source filenames, sizes and row counts as provenance.'),(9,'Summarize schema quality','Record problems before allowing vectorization to continue.'),(12,'Inspect the raw population','These counts include duplicates; they are not modeling counts.')],
    '3_identity_time_audit': [(0,'Measure identity duplication','Check annotation disagreement before selecting a representative record.'),(6,'Define chronological windows','Partition by record timestamps, not filename week strings.'),(13,'Retain canonical identities','Keep the existing earliest-record policy so vector-cache row identities match.'),(15,'Remove later exact-TLSH matches','A similar group must not cross into a later evaluation partition.'),(23,'Check split separation','Check chronology and identity grouping before analyzing features.'),(27,'Export the retained population','Report the effect of the exact-match controls.'),(31,'Plot population and collection time','Keep this figure complete inside one cell so it cannot render halfway through.')],
    '4_label_analysis': [(0,'Declare the role aliases','Only explicit aliases map to the proposed roles; do not invent missing labels.'),(1,'Create label masks','Separate unknown behavior annotations from observed negatives.'),(8,'Measure fitting support','Eligibility is based on fitting-period labels only.'),(12,'Inspect annotation coverage','Missing behavior tags are a data-quality limitation, not benign behavior.'),(14,'Export observed tags and families','Family and confidence are descriptive metadata, not vector inputs.'),(19,'Plot label quality','The four panels use fitting data and share one consistent figure.'),(31,'Plot tag prevalence and confidence','These plots describe annotation quality, not classifier performance.'),(36,'Check the label-analysis milestone','Report how much fitting malware lacks behavior annotations.')],
    '5_vectorization': [(0,'Define the static input dictionary','Only the four allowlisted static groups enter the numerical matrix.'),(8,'Identify the input cache','Reuse vectors only when source text, row identity and extractor hashes match.'),(12,'Read or build the vector matrix','A memory-mapped float32 matrix keeps the dataset within a bounded RAM budget.'),(14,'Choose raw-to-vector checks','Select representative source rows across all available formats.'),(16,'Group the checks by source file','Avoid rereading the same file for every checked row.'),(18,'Check raw-to-vector alignment','Regenerate selected rows independently to detect a misaligned cache.'),(21,'Count invalid feature values','Inspect vectors in small batches rather than allocating a full boolean matrix.'),(22,'Purge later identical vectors','Use no labels when enforcing this conservative duplicate control.'),(28,'Export semantic feature names','Keep categorical byte fields distinct from continuous measurements.'),(32,'Inspect eligible partition sizes','These are the records left after all vector-level controls.'),(34,'Save vectorization provenance','Record input shape, cache identity and independent-check coverage.'),(36,'Check histogram normalization','Frequency groups should sum to one; no classifier is involved.')],
    '6_feature_statistics': [(0,'Compute exact fitting statistics','Process 16 columns at a time over every eligible fitting record.'),(2,'Export feature-quality tables','Identify constants and long tails using fitting data only.'),(8,'Plot fitting distributions','Use the same transforms and axes for each fitting format.'),(17,'Check the feature-statistics milestone','Constants are reported, not silently discarded from the source matrix.')],
    '7_normalization_redundancy': [(0,'Select a reproducible fitting sample','Correlation uses a hash sample; normalization statistics still use all fitting rows.'),(5,'Define fitting-only normalization','Use fitting medians, means and scales; keep the source cache unchanged.'),(8,'Inspect feature redundancy','Correlation is a sample-based association, not evidence of causality.'),(15,'Compare raw and transformed size','Log scaling and standardization are diagnostics, not selected preprocessing winners.'),(23,'Check normalization boundaries','No validation or held-out values determine the normalization statistics.')],
    '8_feature_selection': [(0,'Choose observed fitting targets','Skip unsupported roles; category comparisons use annotated malware only.'),(2,'Allocate univariate score arrays','Keep one score per feature and target without fitting a classifier.'),(4,'Calculate exact ANOVA scores','Use bounded blocks over the complete fitting period.'),(5,'Build an equal-target ranking','Percentile ranks prevent a common role from dominating through raw score magnitude.'),(9,'Estimate mutual information','Use a smaller fixed fitting sample because MI estimation is more expensive.'),(17,'Export the feature rankings','These associations are exploratory and have not been classifier-validated.'),(20,'Build a redundancy-pruned candidate','Retain a representative high-ranked feature from strongly correlated groups.'),(24,'Save proposed feature subsets','Top-64/128/256 subsets are candidates for a later controlled experiment.'),(28,'Inspect the top-ranked inputs','Use semantic names so each proposed input can be reviewed.'),(29,'Plot univariate rankings','Compare the two association measures without treating either as accuracy.'),(34,'Plot ranked-feature correlations','Show where high-ranked inputs still carry overlapping information.'),(43,'Check candidate subsets','Verify constants are excluded while keeping the decision provisional.')],
    '9_pca_visualization': [(0,'Fit PCA on fitting inputs','Labels never enter the decomposition.'),(5,'Export projection evidence','Retain variance and loading tables, not a serialized estimator.'),(9,'Plot the PCA projection','Color the same points by format and label; visual separation is not accuracy.'),(21,'Check the PCA milestone','Variance retention describes this fitting sample, not future predictive performance.')],
    '10_distribution_drift': [(0,'Sample the validation period','Choose records by hash without inspecting their labels.'),(4,'Measure feature-distribution changes','KS distance is an effect size here, not an IID significance claim.'),(5,'Export the drift ranking','Keep temporal drift separate from feature-selection scores.'),(8,'Plot the largest changes','These views are descriptive; they do not tune against held-out labels.'),(14,'Check drift-analysis boundaries','Only fitting and validation inputs enter these comparisons.')],
    '11_findings_reference': [(0,'Read the historical score reference','Read protected score tables only; do not load or recreate a classifier.'),(4,'Assemble the analysis summary','Keep the data, feature and reference facts together for the written findings.'),(7,'Write concise findings','The detailed numerical evidence remains in the exported tables.'),(10,'Check the analysis-only contract','Stop at data understanding; do not advance to supervised modeling.')],
}
MARKDOWN = {
 '1_scope_inventory': '# EMBER2024: primary data and feature analysis\n\n- **This milestone:** understand the inputs and propose feature subsets. No classifier training or evaluation.\n- Analyze the files actually present: currently Win64 train/test features plus the challenge set.\n- Use fitting data for normalization, feature selection and PCA. Historical scores are references only.',
 '2_ingestion':'## 2. Ingest and inspect the raw data\n\nRead every JSONL record once. Export schema, missingness and source audits before proceeding. Raw counts still include duplicates.',
 '3_identity_time_audit':'## 3. Check identities and time boundaries\n\n- Report duplicate identities and annotation disagreements.\n- Split official training into fitting, validation and calibration periods.\n- Purge later exact TLSH matches. Fuzzy and family-disjoint grouping remain future work.',
 '4_label_analysis':'## 4. Understand the labels\n\n- Missing malware tags mean **unknown**, not verified negatives.\n- Map explicit aliases only; check role support on fitting data.\n- Family/confidence fields describe annotation quality and stay outside the input vectors.',
 '5_vectorization':'## 5. Convert static features to inputs\n\nUse the four shared thrember groups: general information, byte histogram, byte entropy and strings. Verify cache identity and raw-to-vector alignment. The 696-input representation omits PE-specific groups.',
 '6_feature_statistics':'## 6. Inspect feature quality\n\nCalculate exact fitting statistics in small column blocks. Look for missing values, constants, sparsity and long tails before choosing transforms.',
 '7_normalization_redundancy':'## 7. Check scales and redundancy\n\nNormalization uses full fitting statistics. Correlations use a reproducible 20,000-row fitting sample and can be sensitive to outliers.',
 '8_feature_selection':'## 8. Propose feature subsets\n\n- Full-fitting ANOVA and sampled MI measure association, not predictive accuracy.\n- Treat unobserved tags as unknown and skip unsupported roles.\n- Export candidate subsets; choosing a winner requires a later supervised experiment.',
 '9_pca_visualization':'## 9. Explore variance with PCA\n\nFit PCA on normalized fitting inputs only. Labels are color overlays, never fitting inputs. Explained variance and visible clusters are not classifier scores.',
 '10_distribution_drift':'## 10. Inspect temporal drift\n\nCompare fitting and validation feature samples. KS distance is descriptive here; related files and time dependence limit IID interpretations. Do not tune from held-out labels.',
 '11_findings_reference':'## 11. Review findings and historical scores\n\nRead the preserved baseline table without loading models. Its earlier APK/PDF/Win64 population differs from the current Win64 analysis. Stop here to review the next proposal step.',
}
PLOT_HELPERS={
 ('4_label_analysis',19):('plot_label_quality','meta, fit, coverage, Y, mal_annotated, LABELS, savefig'),
 ('6_feature_statistics',0):('fitting_statistics','X, idx, D, names'),
 ('6_feature_statistics',8):('plot_feature_distributions','meta, idx, X, groups, stats, savefig'),
 ('9_pca_visualization',9):('plot_pca_structure','explained, coordinates, meta, sample_idx, y, savefig'),
}


def format_findings(summary, mi_sample_size):
    """Keep the notebook's conclusion short; full evidence stays in its tables."""
    q=summary['vector_quality']; duplicate=summary['duplicate_audit']
    unsupported=', '.join(summary['unsupported_roles']) or 'none at the stated threshold'
    return f'''# Primary analysis findings

- **Population:** {summary['raw_rows']:,} raw rows; {summary['eligible_rows']:,} eligible records; {summary['fitting_rows']:,} fitting records ({', '.join(summary['fitting_formats'])}).
- **Inputs:** {summary['features']} named features, including {summary['constant_features']} constant fitting column(s); {q['nonfinite_cells']:,} non-finite cells. Independent vector checks: {q['independent_row_checks']} rows.
- **Labels:** {summary['unannotated_fitting_malware']:,} fitting malware records lack behavior annotations. Unsupported roles: {unsupported}.
- **Identity quality:** {duplicate['binary_conflict_identities']:,} duplicate identities disagree on binary labels; {duplicate['behavior_conflict_identities']:,} disagree on behavior tags.
- **Exploratory selection:** ANOVA uses all fitting rows; MI uses {mi_sample_size:,}. The top-64/128/256 and correlation-pruned sets are proposals, not validated winners.
- **Redundancy and PCA:** {summary['correlation_pairs_098']:,} sample feature pairs have |r| ≥ 0.98. PCA needs {summary['PCA_components_needed']['0.95']} components for 95% of sampled standardized variance.
- **Scope:** historical baseline scores are references only. No classifier was trained or evaluated.

## Review before modeling

1. Keep shared inputs, or inspect Win64-specific PE groups next?
2. Are the role aliases and missing-annotation assumptions defensible?
3. Which candidate subsets and transforms merit a controlled comparison?
4. Should similarity controls include fuzzy/family grouping, and what new evaluation period is available?
'''


def _fragment(source, nodes):
    # AST boundaries split top-level tasks without cutting a loop or expression.
    return '\n'.join(ast.get_source_segment(source,node) for node in nodes)


def _mapping_parts(source, title, comment):
    """Split a long summary literal while preserving key order and values."""
    nodes=ast.parse(source).body
    output=[]
    for node in nodes:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='summary' and isinstance(node.value,ast.Dict):
            d=node.value
            for start in range(0,len(d.keys),8):
                piece=ast.Dict(keys=d.keys[start:start+8],values=d.values[start:start+8])
                statement=ast.Assign(targets=[ast.Name(id='summary',ctx=ast.Store())],value=piece) if start==0 else ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='summary',ctx=ast.Load()),attr='update',ctx=ast.Load()),args=[piece],keywords=[]))
                labels = {0: 'Record population and analysis scope', 8: 'Record feature and label quality', 16: 'Record projection and reference provenance'}
                output.append((labels[start], comment, ast.unparse(ast.fix_missing_locations(statement))))
        else:
            text=ast.get_source_segment(source,node)
            if output and output[-1][0]==title: output[-1]=(title,comment,output[-1][2]+'\n'+text)
            else:
                label = 'List unsupported roles' if isinstance(node, ast.Assign) else 'Export the analysis summary'
                output.append((label, comment, text))
    return output


def readable_notebook(notebook, kind='analysis'):
    """Return a readable notebook with section IDs shared by its smaller steps."""
    import black
    if notebook.metadata.get('readable_layout_version') == 1:
        return notebook
    cells=[]
    if kind=='download':
        cells.append(nbformat.v4.new_markdown_cell('# Add the pinned Win64 feature data\n\n- Download feature JSONL archives, not executable binaries.\n- Verify archive hashes before extraction; reuse verified files on rerun.\n- Keep official train/test boundaries. The analysis notebook audits duplicates.'))
    for original in notebook.cells:
        if original.cell_type!='code': continue
        stage=original.metadata['stage']; source=original.source; nodes=ast.parse(source).body
        if kind=='analysis':
            cells.append(nbformat.v4.new_markdown_cell(MARKDOWN[stage]))
            plan=PLANS[stage]; groups=[]
            for i,(start,title,comment) in enumerate(plan):
                end=plan[i+1][0] if i+1<len(plan) else len(nodes)
                text=_fragment(source,nodes[start:end])
                if (stage,start) in PLOT_HELPERS:
                    name,args=PLOT_HELPERS[(stage,start)]
                    text=f'from multi_label_static_analysis.notebook_steps import {name}\n'
                    text+=f'stat_records = {name}({args})' if name=='fitting_statistics' else f'{name}({args})'
                if stage=='11_findings_reference' and start==7:
                    text='from multi_label_static_analysis.notebook_readability import format_findings\nnotes = format_findings(summary, len(mi_idx))\n(REPORT / "findings.md").write_text(notes)\ndisplay(Markdown(notes))'
                if stage=='11_findings_reference' and start==4:
                    groups.extend(_mapping_parts(text,title,comment))
                else: groups.append((title,comment,text))
        else:
            if stage=='1_download_setup':
                cells.append(nbformat.v4.new_markdown_cell('## 1. Check sources and storage\n\nRead the pinned revision and checksums. Store temporary downloads inside the project.'))
                # Imports/paths, hub settings, then the disk/provenance checks.
                cuts=[(0,'Load project paths','Keep all acquisition files inside the project.'),(5,'Configure the download cache','Set cache locations before importing the Hub client.'),(12,'Check source revision and capacity','Validate the pinned source and disk capacity before continuing.')]
                groups=[]
                for i,(start,title,comment) in enumerate(cuts):
                    end=cuts[i+1][0] if i+1<len(cuts) else len(nodes)
                    groups.append((title,comment,_fragment(source,nodes[start:end])))
            elif stage=='2_download_extract':
                cells.append(nbformat.v4.new_markdown_cell('## 2. Reuse or extract verified archives\n\nProcess one archive at a time. Hash it before extraction, record each JSONL file, then remove the compressed copy.'))
                groups=[('Open the completion record','A completed record lets reruns avoid network transfer.', '''from multi_label_static_analysis.data_acquisition import process_archive
completed_path = P.artifacts / 'win64_download_manifest.json'
completed = json.loads(completed_path.read_text()) if completed_path.exists() else {'revision': SOURCE['revision'], 'archives': {}}
assert completed['revision'] == SOURCE['revision']'''),('Process the pinned archives','The helper verifies hashes and extracts JSONL members using safe base filenames.', '''for item in sorted(SOURCE['files'], key=lambda f: '_train' not in f['filename']):
    process_archive(item, P, SOURCE, DOWNLOAD, completed, completed_path)
print('PASS: both archives verified and JSONL feature files extracted; completion manifest saved.')''')]
            else:
                cells.append(nbformat.v4.new_markdown_cell('## 3. Verify the extracted data\n\nCheck schema and expected raw-row coverage. Unique-record counts are established later by the analysis audit.'))
                groups=[('Check the feature files','Repeated raw rows are allowed here; the analysis step handles deduplication.',source)]
        for part,(title,comment,text) in enumerate(groups,1):
            formatted=black.format_str(text,mode=black.Mode(line_length=88)).strip()
            # A file-write result is a character count, not an analysis output.
            last=ast.parse(formatted).body[-1]
            if isinstance(last,ast.Expr) and isinstance(last.value,ast.Call) and isinstance(last.value.func,ast.Attribute) and last.value.func.attr=='write_text': formatted+=';'
            cell=nbformat.v4.new_code_cell(f'# {title}\n# {comment}\n\n'+formatted+'\n')
            cell.metadata.update(stage=stage,step=part,step_count=len(groups),title=title,stage_end=part==len(groups))
            cells.append(cell)
    if kind=='analysis':
        cells.append(nbformat.v4.new_markdown_cell('## Historical reference provenance\n\nThe baseline comparison was recovered from the earlier recorded CSV report and checked against the notebook reference table. It was not recomputed. See `results/historical/baseline_score_manifest.json`. Its earlier mixed-format population differs from the current fitting data.'))
    notebook.cells=cells
    notebook.metadata['readable_layout_version'] = 1
    return notebook
