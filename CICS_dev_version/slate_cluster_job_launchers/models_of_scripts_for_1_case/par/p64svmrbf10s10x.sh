#!/usr/bin/env bash

# Options SBATCH :
#SBATCH -A p19classhd ## project_name (mandatory to launch jobs)
#SBATCH --job-name=p64svmrbf10s10x ## job name
#SBATCH -n10
#SBATCH --mail-type=ALL ## to receive all notifications by mail
#SBATCH --mail-user=amad.diouf@inserm.fr ## mail adress to receive notifications
#SBATCH -o ./outputs/p64svmrbf10s10x.%j.logo ## filename for standard output
#SBATCH -e ./outputs/p64svmrbf10s10x.%j.loge ## filename for error output
#SBATCH --time=240:00:00 ## to set the wall time
#SBATCH --partition=long ## to provide the type of partition (short: time-limit=2 days, priority=4 ; middle: time-limit=4 days, priority=3 ; long: time-limit=10 days, priority=2, best: time-limit=unlimited (but job killed and rescheduled if there's a queue), priority=1)

# Jobs Steps (étapes du Job) :

# Step 1 : ## make temporary directory
TMPDIR=$(mktemp "tmp.$SLURM_JOB_ID")

# Step 2 : ## load conda environnement
source /data/diouf/anaconda3/etc/profile.d/conda.sh
conda activate classhd37_env4

# Les k Steps suivants sont les k processus executed in parallel
python SL_prod_offshore_4.py -t "p64svmrbf10s10x" -xproc 10 -sl "Classif" -ca "SVM_Mark2Vpar" -cs "Both" -cla_msn 30 -cla_seeds 10 -cla_profiles_path "PDX__data/processedDataBis/pData.64"  ## python command line

# Step 2 : ## unload conda environnement
conda deactivate