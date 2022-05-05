#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=1              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=4GB                        # CPU memory
#SBATCH --job-name=python-train-audio-zh
#SBATCH --output=log/tts-%j.out
#SBATCH --gres=gpu:1

module purge
singularity exec --nv \
    --overlay /scratch/bh2283/code/zh-audio.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; python train_tacotron2.py"