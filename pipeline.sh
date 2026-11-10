#!/bin/bash
# Waits for COCO extraction, then chains: smoke train -> smoke eval -> full train -> final eval
cd /iitgn/home/syed.tashif/climp
until grep -q UNZIPDONE data/coco/unzip.log; do sleep 60; done
cat data/coco/unzip.log
S=$(sbatch --parsable --job-name=climp_smoke --time=00:40:00 \
     --export=ALL,MAX_STEPS=60,EPOCHS=1,SAVE_EVERY_STEPS=0,OUT_DIR=$HOME/climp/runs/smoke submit_ft.sh)
SE=$(sbatch --parsable --dependency=afterok:$S --job-name=climp_smoke_eval submit_eval.sh runs/smoke/climp_ft_best.pth)
F=$(sbatch --parsable --dependency=afterok:$SE submit_ft.sh)
E=$(sbatch --parsable --dependency=afterok:$F submit_eval.sh runs/v3_ft/climp_ft_best.pth)
echo "smoke=$S smoke_eval=$SE full_train=$F final_eval=$E"
