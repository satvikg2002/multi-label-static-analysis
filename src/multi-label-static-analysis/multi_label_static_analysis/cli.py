"""Separate commands for primary analysis and repository baseline experiments."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import runpy
import sys
from .config import project_paths


def main(argv=None):
    parser=argparse.ArgumentParser(description='EMBER2024 primary analysis and isolated repository baselines')
    parser.add_argument('--project-dir',type=Path)
    commands=parser.add_subparsers(dest='command',required=True)
    execute=commands.add_parser('run',help='Execute the analysis notebook only')
    execute.add_argument('--interactive',action='store_true',help='Pause after each analysis section')
    execute.add_argument('--timeout',type=int,default=7200)
    commands.add_parser('verify',help='Verify analysis outputs, model removal and preserved scores')
    commands.add_parser('baseline-scores',help='Display historical scores without models or retraining')
    baseline=commands.add_parser('run-baseline',help='Execute the new binary and native-tag baseline notebook')
    baseline.add_argument('--interactive',action='store_true',help='Pause after each baseline step')
    baseline.add_argument('--timeout',type=int,default=7200)
    commands.add_parser('verify-baseline',help='Verify the completed repository baseline experiment')
    baseline_build=commands.add_parser('build-baseline',help='Generate the repository baseline notebook')
    baseline_build.add_argument('--output',type=Path)
    baseline_build.add_argument('--force',action='store_true')
    commands.add_parser('download-win64',help='Run the existing pinned data acquisition notebook')
    build=commands.add_parser('build',help='Generate a fresh analysis notebook')
    build.add_argument('--output',type=Path); build.add_argument('--force',action='store_true')
    args=parser.parse_args(argv); paths=project_paths(args.project_dir)
    if args.command in {'run','run-baseline'}:
        if args.timeout<=0: parser.error('--timeout must be positive')
        from .workflow import run_notebook
        execution_paths=paths
        if args.command=='run-baseline':
            execution_paths=replace(paths,
                notebook=paths.project/'notebooks/ember2024_repository_baseline.ipynb',
                reports=paths.project/'results/repository_baseline')
        run_notebook(execution_paths,interactive=args.interactive,timeout=args.timeout)
    elif args.command=='verify':
        from .workflow import verify_outputs
        verify_outputs(paths)
    elif args.command=='baseline-scores':
        from .workflow import baseline_scores
        print('Historical reference only. Previous trained models have been removed.')
        print(baseline_scores(paths).to_string(index=False))
    elif args.command=='verify-baseline':
        from .repository_baseline import verify_baseline
        verify_baseline(paths)
    else:
        previous_cwd,previous_argv=Path.cwd(),sys.argv
        try:
            os.chdir(paths.root)
            if args.command=='download-win64':
                runpy.run_module('multi_label_static_analysis.notebook_builders.build_download_notebook',run_name='__main__')
                from .workflow import run_notebook
                run_notebook(replace(paths,notebook=paths.project/'notebooks/ember2024_win64_download.ipynb'),require_data=False)
            else:
                default_output=paths.project/'notebooks/ember2024_repository_baseline.ipynb' if args.command=='build-baseline' else paths.notebook
                output=(paths.root/args.output).resolve() if args.output else default_output
                if output.exists() and not args.force: parser.error('Output exists; choose --output or explicitly use --force.')
                sys.argv=['build_analysis_notebook','--output',str(output)]
                if args.force: sys.argv.append('--force')
                module='build_repository_baseline' if args.command=='build-baseline' else 'build_analysis_notebook'
                runpy.run_module('multi_label_static_analysis.notebook_builders.'+module,run_name='__main__')
        finally:
            os.chdir(previous_cwd); sys.argv=previous_argv
