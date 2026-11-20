"""Stable, serializable helpers for the research notebook's fitted pipelines."""
def column_subset(values, columns):
    """Select the fixed static feature-group columns declared by the pipeline."""
    return values[:, columns]
