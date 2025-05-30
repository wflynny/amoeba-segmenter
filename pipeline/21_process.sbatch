#!/usr/bin/env bash
### SLURM HEADER
#SBATCH --job-name=segment
#SBATCH --output=/dev/null

#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB

#SBATCH --export=ALL
### SLURM HEADER

localcores=${SLURM_CPUS_PER_TASK}

eval "$(conda shell.bash hook)"
conda activate amoeba-segment

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Something wrong" >&2
    exit 2
fi

if [ -z "$AMOEBASEGMENTER_LOADED" ]; then
    echo "Need to source global config" >&2
    exit 123
fi

tiff_file="${1}_${SLURM_ARRAY_TASK_ID}.ome.tiff"

python ${AMOEBASEGMENTER_SCRIPTS_DIR}/process.py "${tiff_file}" "${2}"
