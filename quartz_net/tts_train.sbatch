#!/bin/bash
#SBATCH --nodes=1                        # requests 3 compute servers
#SBATCH --ntasks-per-node=8              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=20:00:00
#SBATCH --mem=20GB                        # CPU memory
#SBATCH --job-name=python-tts-chinese
#SBATCH --output=log/tts-%j.out
#SBATCH --gres=gpu:rtx8000:1

module purge
singularity exec --nv \
    --overlay /scratch/bh2283/senv/torchaudio-python3.10.ext3:ro \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
    /bin/bash -c "source /ext3/env.sh; python dur_fft_taco2.py"