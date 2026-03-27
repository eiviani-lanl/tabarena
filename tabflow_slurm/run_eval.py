from pathlib import Path
from simple_evaluation.run_eval_for_new_model import run_eval_for_new_models, ModelMetadata

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on cached optimization")
    parser.add_argument(
        "--type",
        type=str,
        default="all",
        help="Dataset type to evaluate",
    )
    
    type = parser.parse_args().type

    suff = "" if type=="all" else f"_{type}"

    models = [ModelMetadata(
                path_raw=Path(f"/vast/home/eiviani/output/elm_experiment_040326{suff}"),
                method="CustomELM",
                new_result_prefix="new_",
                only_load_cache=False,
            )]

    fig_dir = Path(__file__).parent / "evals" / "custom_elm" / type
    print("FIGS:",fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_eval_for_new_models(
        models,
        fig_output_dir=fig_dir,
        extra_subsets=[["lite"]],
    )
