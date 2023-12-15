#!/bin/bash

#SBATCH --job-name=RL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=25GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu

# job info
exp_id=$1
seed=$2

# Singularity path
ext3_path=/scratch/$USER/my_env/py10/overlay-25GB-500K.ext3
sif_path=/scratch/$USER/my_env/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
cd /scratch/$USER/NYU/RL/Project
python -m algos.td3 --exp_id ${exp_id} --seed ${seed}
"