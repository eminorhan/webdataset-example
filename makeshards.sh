#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
##SBATCH --gres=gpu:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=makeshards
#SBATCH --output=makeshards_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch5/LakeGroup/shared_data/webdataset-examples/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/Y' --dataset-name 'SAY_1fps_300'

echo "Done"
