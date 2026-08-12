import os

import glob
from pathlib import Path
from simple_evaluation.run_eval_for_new_model import run_eval_for_new_models, ModelMetadata
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata
import argparse
import yaml
import pandas as pd
import shutil
from tabarena.loaders import Paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on cached optimization")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["binary","multiclass","regression","all"],
        help="Dataset type to evaluate",
    )
    parser.add_argument(
        "--direct_links",
        type=str,
        default="both",
        choices=["with","without","both"],
        help="Evaluate with or without direct links, or both",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="elm_experiment_081126",
        help="`benchmark_name` set in `run_setup_slurm_jobs.py`",
    )

    parser.add_argument(
        "--methods",
        nargs='+',
        type=str,
        default="GFDL",
        help="methods to be evaluated, defaults to gfdl",
    )
    
    dataset = parser.parse_args().dataset
    dl = parser.parse_args().direct_links
    exp = parser.parse_args().name

    meths = parser.parse_args().methods

    meths = [meths] if type(meths)==str else meths

    print(meths)

    suff = "" if dataset=="all" else f"_{dataset}"

    scheme = "" if dl=="both" else f"/{dl}_direct_links"

    config_path = Path(f"/vast/home/eiviani/tabarena/tabflow_slurm/benchmark_configs_{exp}{suff}.yaml")
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)

    base_path = Path(f"/vast/home/eiviani/output/{exp}{suff}")
    direct_links_base = base_path / "with_direct_links"
    no_direct_links_base = base_path / "without_direct_links"

    direct_links_base.mkdir(exist_ok=True)
    no_direct_links_base.mkdir(exist_ok=True)

    direct_links_dir = direct_links_base / "data"
    no_direct_links_dir = no_direct_links_base / "data"

    direct_links_dir.mkdir(exist_ok=True)
    no_direct_links_dir.mkdir(exist_ok=True)

    def if_new(src, dst):
        if not os.path.exists(dst) or os.stat(src).st_mtime > os.stat(dst).st_mtime:
            shutil.copy2(src, dst)
            print(f"added {src} to {dst}")

    runs = glob.glob(f"{base_path}/data/*")

    for cls in meths:
        c_path = base_path / cls
        c_path.mkdir(exist_ok=True)
        data_path = c_path / "data"
        data_path.mkdir(exist_ok=True)

        cls_runs = glob.glob(f"{base_path}/data/{cls}*")

        for file in cls_runs:
            name = Path(file).name
            dst = data_path / name
            shutil.copytree(file, dst, copy_function=if_new, dirs_exist_ok=True)

    for config in all_configs['methods']:
        cls = config.get('model_cls')

        if cls not in ['gfdl','CELM','sELM']: continue

        method_kwargs = config.get('method_kwargs', {})
        model_hyperparams = config.get('model_hyperparameters', {})
        has_direct_links = model_hyperparams.get('direct_links', False)
        
        data = Path(config.get('name', ""))
        
        if has_direct_links:
            dest = direct_links_dir / data
        else:
            dest = no_direct_links_dir / data
        
        try:
            shutil.copytree(base_path / "data" / data, dest, copy_function=if_new, dirs_exist_ok=True)
        except:
            continue


    models = [
            ModelMetadata(
                path_raw=Path(f"/vast/home/eiviani/output/{exp}{suff}{scheme}/{cls}"),
                method=cls,
                only_load_cache=False,
            ) for cls in meths]
    
    print(models)
    fig_dir = Path(__file__).parent / "evals" / "custom_elm" / dataset / Path("" if dl=="both" else dl)
    print("FIGS:",fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = run_eval_for_new_models(
        models,
        fig_output_dir=fig_dir,
        extra_subsets=[["lite"]],
    )
    
    df = pd.concat(df)

    config_results = (
        df.groupby(["dataset", "fold", "method"], as_index=False)
        .mean(numeric_only=True)
    )

    best_idx = (
        config_results
        .groupby(["dataset", "fold"])["metric_error_val"]
        .idxmin()
    )

    selected = config_results.loc[best_idx].sort_values(
        ["dataset", "fold"]
    )
    win_counts = (
        selected.groupby(["dataset", "method"])
        .size()
        .rename("num_folds_selected")
        .reset_index()
    )

    winning_configs = (
        win_counts.sort_values(
            ["dataset", "num_folds_selected"],
            ascending=[True, False],
        )
        .groupby("dataset", as_index=False)
        .first()
    )

    print(winning_configs)

    for config in all_configs['methods']:
        method_kwargs = config.get('method_kwargs', {})
        model_hyperparams = config.get('model_hyperparameters', {})
        has_direct_links = model_hyperparams.get('direct_links', False)
        
        data = Path(config.get('name', ""))
        if str(data) in list(winning_configs['method']):
            print(data)
            print(model_hyperparams)
