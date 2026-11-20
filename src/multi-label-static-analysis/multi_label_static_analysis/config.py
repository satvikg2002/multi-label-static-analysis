"""Project locations shared by the CLI, scripts, and research notebook."""
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    project: Path
    notebook: Path
    data: Path
    reports: Path
    artifacts: Path
    historical: Path


def project_paths(start: str | Path | None = None) -> ProjectPaths:
    """Find this uv project from a working directory or explicit project path."""
    location = Path(start or Path.cwd()).resolve()
    if location.is_file():
        location = location.parent
    for root in (location, *location.parents):
        config_file = root / 'pyproject.toml'
        if not config_file.is_file():
            continue
        config = tomllib.loads(config_file.read_text())
        if config.get('project', {}).get('name') != 'multi-label-static-analysis':
            continue
        settings = config.get('tool', {}).get('multi-label-static-analysis', {})
        base = 'src/multi-label-static-analysis/' if (root/'src/multi-label-static-analysis').is_dir() else ''
        def resolve(key: str, default: str) -> Path:
            return (root / settings.get(key, default)).resolve()
        return ProjectPaths(
            root=root,
            project=(root/base).resolve(),
            notebook=resolve('notebook', base+'notebooks/ember2024_primary_analysis.ipynb'),
            data=resolve('data-dir', base+'multi_label_static_analysis/data/ember_data'),
            reports=resolve('reports-dir', base+'results/primary_analysis'),
            artifacts=resolve('artifacts-dir', base+'artifacts'),
            historical=resolve('historical-reports-dir', base+'results/historical'),
        )
    raise FileNotFoundError('Cannot find this project’s pyproject.toml. Run inside the project or pass --project-dir.')
