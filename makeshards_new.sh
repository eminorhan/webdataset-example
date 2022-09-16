#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=makeshards_new
#SBATCH --output=makeshards_new_%A_%a.out

module purge
module load cuda-11.4

# sayavakepicutego4d
python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards_new.py --cache-path '/misc/vlgscratch5/LakeGroup/emin/webdataset-example/sayavakepicutego4d.pth' --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/ava_test' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/ava_trainval' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/kcam' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P01' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P02' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P03' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P04' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P06' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P07' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P09' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P11' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P12' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P22' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P23' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P25' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P26' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P27' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P28' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P30' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P33' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P34' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P35' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P36' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P37' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/UT_ego' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/Y' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/ego4d' --dataset-name 'sayavakepicutego4d' --shards '/misc/vlgscratch4/LakeGroup/emin/' --data-fraction 0.1 --seed 1

echo "Done"
