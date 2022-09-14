#!/bin/bash
#SBATCH --account=astro
#SBATCH --partition=astro
#SBATCH --job-name=galplane
#SBATCH --chdir=/gscratch/scrubbed/lynnej/galplane/sims_featureScheduler_runs2.2/galplane

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --time=36:00:00
#SBATCH --mem=500G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lynnej@uw.edu
#SBATCH --export=all

source ~/loadmods

source ~/gg/miniconda3/etc/profile.d/conda.sh
export RUBIN_SIM_DATA_DIR='/gscratch/astro/lynnej/rubin_sim_data'
conda activate rubin

export OPENBLAS_NUM_THREADS=1


parallel:: 'python galplane_priority.py --agg_level 1.5' 'python galplane_priority.py --agg_level 2.0'

parallel ::: scimaf_dir metadata_dir ddf_dir glance_dir
