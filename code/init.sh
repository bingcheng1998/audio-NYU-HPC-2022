module purge
module load anaconda3/2020.07
source activate /home/bh2283/penv/
du -hs ./*
du -hs .[^.]*
myquota

singularity exec --overlay zh-audio.ext3 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif /bin/bash

singularity exec \
--overlay /scratch/bh2283/senv/torchaudio-python3.10.ext3:ro \
--overlay /scratch/bh2283/data/data_aishell3.sqf:ro \
--overlay /scratch/bh2283/data/aidatatang_200zh.sqf:ro \
--overlay /scratch/bh2283/data/ST-CMDS-20170001_1-OS.sqf:ro \
 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
 -c "source /ext3/env.sh; python train_batch.py"


 mksquashfs imagenet-example imagenet-example.sqf  -keep-as-directory

 singularity exec \
--overlay /scratch/bh2283/senv/torchaudio-python3.10.ext3:ro \
--overlay /scratch/bh2283/data/data_aishell3.sqf:ro \
 /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif \
/ext3/miniconda3/bin/python

mksquashfs ST-CMDS-20170001_1-OS /scratch/bh2283/data/ST-CMDS-20170001_1-OS.sqf  -keep-as-directory

用tar命令批量解压某个文件夹下所有的tar.gz文件
ls *.tar.gz | xargs -n1 tar xzf

find aidatatang_200zh -type d -exec chmod 755 {} \;
find aidatatang_200zh -type f -exec chmod 644 {} \;
mksquashfs aidatatang_200zh /scratch/bh2283/data/aidatatang_200zh.sqf  -keep-as-directory