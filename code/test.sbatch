#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=0:10:00
#SBATCH --mem=1GB                        # CPU memory
#SBATCH --job-name=python-train-audio-zh
#SBATCH --output=log/audio-zh-%j.out
#SBATCH --gres=gpu:rtx8000:0

module purge
singularity exec --nv \
    --overlay /scratch/bh2283/code/zh-audio2.ext3:ro \
    --overlay /scratch/bh2283/data/aidatatang_200zh.sqf:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash
    # -c "source /ext3/env.sh; python utils/dataset.py"