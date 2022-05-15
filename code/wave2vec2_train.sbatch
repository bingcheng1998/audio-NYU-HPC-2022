#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=23:00:00
#SBATCH --mem=8GB                        # CPU memory
#SBATCH --mail-type=END
#SBATCH --mail-user=bh2283@nyu.edu
#SBATCH --job-name=wav2vec-zh-audio
#SBATCH --output=log/wav2vec-zh-%j.out
#SBATCH --gres=gpu:rtx8000:1

module purge
singularity exec --nv \
    --overlay /scratch/bh2283/code/zh-audio.ext3:ro \
    --overlay /scratch/bh2283/data/ST-CMDS-20170001_1-OS.sqf:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; python train_wav2vec2.py"