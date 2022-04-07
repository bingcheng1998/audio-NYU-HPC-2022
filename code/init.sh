module purge
module load anaconda3/2020.07
source activate /home/bh2283/penv/
du -hs ./*
du -hs .[^.]*
myquota
