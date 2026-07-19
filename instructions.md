## TabArena (ELM Extension Fork)

### A Living Benchmark for Tabular ML with ELM Integration

---

| [Upstream Leaderboard](https://tabarena.ai/) | [Examples](https://tabarena.ai/code-examples) | [Paper](https://tabarena.ai/paper-tabular-ml-iid-study) |

---

Instructions for running the TabArena benchmarking system with support for our ELM.

Adapted from TabArena's [README.md](https://github.com/autogluon/tabarena/blob/main/README.md)

Fork + branch:
https://github.com/eiviani-lanl/tabarena/tree/eiviani_elm

---

# Installation

To install TabArena, ensure you are using Python 3.9-3.12. Then, run the following:

### Install UV

Ensure [UV is installed](https://docs.astral.sh/uv/getting-started/installation/) for the most stable install.

```
pip install uv  # if pip is available
```

### Install AutoGluon

In future AutoGluon installation will occur automatically, but due to changes yet to be released, we need to install AutoGluon from source.

```
git clone https://github.com/autogluon/autogluon.git
./autogluon/full_install.sh
```

### Clone the repository

```
git clone https://github.com/eiviani-lanl/tabarena.git
cd tabarena
git checkout eiviani_elm
```

### Benchmarking our ELM

```
uv pip install --prerelease=allow -e "./tabarena[benchmark]"
```
---
# Model

The ELM wrapped in autogluon is contained in `tabarena/tabarena/benchmark/models/ag/elm`.

---

# Running experiments

Most of the code you will work with will be contained in either `tabarena\examples` or in `tabarena\tabflow_slurm`.

## For HPO on a cluster:
`cd tabflow_slurm`
### Code Overview

* `benchmarking_setup/` - contains the code to set up benchmarking on SLURM (see its README for more details).
* `run_tabarena_experiment.py` - contains code to run an individual experiment. This is the code that is run in one jobs
  on the SLURM cluster.
* `run_setup_slurm_jobs.py` - contains code to set up jobs we want to submit to the SLURM cluster (checking duplicates,
  which jobs have to be run, etc.). Also see `setup_slurm_base.py` for more details.
* `submit_template.sh` - contains the array job template for our SLURM jobs and is called/executed after running
  `run_setup_slurm_jobs.py`.
* `simple_evaluation` - contains code to evaluate the results of the experiments.

### Run

Ensure internet access!
Run all code contained in `./benchmarking_setup/`:
```
python ./benchmarking_setup/*.py
```
You may need to add TabArena to the python path in the case that you encounter `ModuleNotFoundError: No module named 'tabflow_slurm'`

You can adjust the search space of the ELM by altering `tabarena/tabarena/tabarena/models/gfdl/generate.py`

### All datasets:

Run `run_setup_slurm_jobs.py` to set up all data needed to run array jobs and submit the array job to the slurm
cluster by following the printed instructions.

NOTE: You'll have to manually edit the options in `run_setup_slurm_jobs.py` according to your local SLURM setup (partitions, paths, etc.)

Not sure if this is an issue with my environment, but they're using `yaml.safe_load()` to load the generated yaml file but the yaml file contains python specific tags, `!!python/tuple`. As such, you will probably have to `grep` to remove all `!!python/tuple` tags.

### Per dataset type:

Run `run_setup_slurm_jobs_by_type.py` to set up all data needed to run array jobs and submit the array job to the slurm
cluster by following the printed instructions. You should recieve 3 separate commands; one per dataset type.

NOTE: You'll have to manually edit the options in `run_setup_slurm_jobs_by_type.py` according to your local SLURM setup (partitions, paths, etc.)

Not sure if this is an issue with my environment, but they're using `yaml.safe_load()` to load the generated yaml file but the yaml file contains python specific tags, `!!python/tuple`. As such, you will probably have to `grep` to remove all `!!python/tuple` tags.

## Evaluation

Run `python run_eval.py --type {all|binary|multiclass|regression} --direct_links {both|with|without} --name`

The name flag denotes the name of the experiment as set in `run_setup_slurm_jobs.py`

The generated figures will reside in `evals/custom_elm/{all|binary|multiclass|regression}` or, if you set the direct_links flag: `evals/custom_elm/{all|binary|multiclass|regression}/{with/without}`

To compare the performance of the ELM and RVFL, you must first run `python run_eval.py --direct_links with` AND `python run_eval.py --direct_links without`. This forces Tabarena to run it's evaluation on the two methods separately.

After running both of the above commands, run `python comparison.py` to access some basic statistics comparing ELM and RVFL to the rest of the methods. You can view the logic in this file if you wish to perform further data analysis.
