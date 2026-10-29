#!/bin/bash
# Slurm job for one ClearMap pipeline on one experiment folder.
#
#   sbatch --export=ALL,CLEARMAP_PIPELINE=tube_map,CLEARMAP_EXPERIMENT=/path/to/experiment clearmap_job.sbatch
#
# Adapt the resources (--cpus-per-task, --mem, --time, --partition and --gres for the GPU of deep_fill) and the
# way the conda environment is activated to your cluster.  The lines below are a starting point, not a recipe.

#SBATCH --job-name=clearmap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=clearmap_%j.out
#SBATCH --error=clearmap_%j.err

set -euo pipefail

: "${CLEARMAP_PIPELINE:?set CLEARMAP_PIPELINE to align, tube_map or cell_map}"
: "${CLEARMAP_EXPERIMENT:?set CLEARMAP_EXPERIMENT to the experiment folder}"
CLEARMAP_CONDA_ENV="${CLEARMAP_CONDA_ENV:-ClearMap3.1}"

# The libraries (OpenMP, BLAS) start one thread per core of the NODE unless told otherwise: tell them the cores of the JOB
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Temporary files: hundreds of GB, so the local disk of the node (or a scratch folder), never a default of /tmp.
# CLEARMAP_TMP, else the TMPDIR of the job when the cluster provides one; ClearMap.run stops without a valid folder
# (it also reads the temp_folder of the machine config, which the variable overrides)
CLEARMAP_TMP="${CLEARMAP_TMP:-${TMPDIR:-}}"
: "${CLEARMAP_TMP:?set CLEARMAP_TMP (or TMPDIR) to a folder of the node with room for hundreds of GB}"
export CLEARMAP_TMP
export QT_QPA_PLATFORM=offscreen  # in case something imports Qt: no display on a compute node

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CLEARMAP_CONDA_ENV}"

srun python -m ClearMap.run "${CLEARMAP_PIPELINE}" "${CLEARMAP_EXPERIMENT}"
