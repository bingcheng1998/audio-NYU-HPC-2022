module purge
module load anaconda3/2020.07
source activate /home/bh2283/penv/
du -hs ./*
du -hs .[^.]*
myquota

singularity exec --overlay zh-audio.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash