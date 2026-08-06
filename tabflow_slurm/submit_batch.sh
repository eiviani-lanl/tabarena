#!/bin/bash

set -euo pipefail

START=${1:-0}
FINAL=815
CHUNK_SIZE=98

if (( START > FINAL )); then
    echo "All array tasks have been submitted."
    exit 0
fi

END=$((START + CHUNK_SIZE - 1))
if (( END > FINAL )); then
    END=$FINAL
fi

echo "Submitting array tasks ${START}-${END}"

ARRAY_JOB_ID=$(
    sbatch --parsable \
        --array="${START}-${END}%${CHUNK_SIZE}" \
        --partition=shared-gpu \
        --cpus-per-task=8 \
        --mem-per-cpu=4G \
        --output="/vast/home/eiviani/slurm_out/elm_experiment_073026/%A/slurm-%A_%a.out" \
        /vast/home/eiviani/tabarena/tabflow_slurm/submit_template.sh \
        /vast/home/eiviani/tabarena/tabflow_slurm/slurm_run_data_elm_experiment_073026.json
)

echo "Submitted array job ${ARRAY_JOB_ID}"

NEXT_START=$((END + 1))

if (( NEXT_START <= FINAL )); then
    NEXT_JOB_ID=$(
        sbatch --parsable \
            --dependency="afterany:${ARRAY_JOB_ID}" \
            "$0" "${NEXT_START}"
    )

    echo "Queued controller ${NEXT_JOB_ID} for tasks ${NEXT_START}-${FINAL}"
fi