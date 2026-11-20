# Primary analysis findings

- **Population:** 1,286,315 raw rows; 646,079 eligible records; 400,000 fitting records (Win64).
- **Inputs:** 696 named features, including 1 constant fitting column(s); 0 non-finite cells. Independent vector checks: 22 rows.
- **Labels:** 101,312 fitting malware records lack behavior annotations. Unsupported roles: Botnet Agent.
- **Identity quality:** 0 duplicate identities disagree on binary labels; 0 disagree on behavior tags.
- **Exploratory selection:** ANOVA uses all fitting rows; MI uses 6,000. The top-64/128/256 and correlation-pruned sets are proposals, not validated winners.
- **Redundancy and PCA:** 76 sample feature pairs have |r| ≥ 0.98. PCA needs 308 components for 95% of sampled standardized variance.
- **Scope:** historical baseline scores are references only. No classifier was trained or evaluated.

## Review before modeling

1. Keep shared inputs, or inspect Win64-specific PE groups next?
2. Are the role aliases and missing-annotation assumptions defensible?
3. Which candidate subsets and transforms merit a controlled comparison?
4. Should similarity controls include fuzzy/family grouping, and what new evaluation period is available?
