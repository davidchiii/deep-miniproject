#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=70GB
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name=torch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=changdavid52@gmail.com

module purge

singularity exec --nv \
            --overlay /scratch/dc4488/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python param_tuning_epoch.py"
