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

# sayavakepicutego4d
python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/ava_test' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/ava_trainval' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/kcam' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P01' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P02' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P03' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P04' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P06' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P07' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P09' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P11' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P12' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P22' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P23' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P25' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P26' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P27' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P28' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P30' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P33' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P34' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P35' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P36' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/P37' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/UT_ego' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/Y' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/ego4d' --dataset-name 'sayavakepicutego4d' --shards '/misc/vlgscratch2/LakeGroup/emin/'

# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/A' --dataset-name 'A_1fps_300s'
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/Y' --dataset-name 'Y_1fps_300s'
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/S' --dataset-name 'S_1fps_300s'
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/A' --dataset-name 'A_5fps_300s'
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/Y' --dataset-name 'Y_5fps_300s'
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/S' --dataset-name 'S_5fps_300s'

# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/1fps_300s/Y' --dataset-name 'SAY_1fps_300s'

# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/Y' --dataset-name 'SAY_5fps_300s'

# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/val' --dataset-name 'imagenet_val'

## ECOSET
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/shared_data/ecoset/train' --dataset-name 'ecoset_train' --maxcount 1e6
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/shared_data/ecoset/val' --dataset-name 'ecoset_val' --maxcount 1e6
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/shared_data/ecoset/test' --dataset-name 'ecoset_test' --maxcount 1e6

## ImageNet
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/train' --dataset-name 'imagenet_train' --maxcount 1e6
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch4/LakeGroup/emin/robust_vision/imagenet/val' --dataset-name 'imagenet_val' --maxcount 1e6

## Places365
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch5/LakeGroup/shared_data/places365/train_large_places365standard' --dataset-name 'places_train' --maxcount 1e6
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards_places_val.py --data-dirs '/misc/vlgscratch5/LakeGroup/shared_data/places365/val_large' --dataset-name 'places_val' --maxcount 1e6

## Labeled S
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards_split.py --data-dirs '/misc/vlgscratch4/LakeGroup/shared_data/S_clean_labeled_data_1fps_5' --dataset-name 'labeled_s' --maxcount 1e6

## KOnkle objects
# python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards_split.py --data-dirs '/misc/vlgscratch5/LakeGroup/emin/konkle_objects' --dataset-name 'konkle_objects' --maxcount 1e6

## Core50
#python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch2/LakeGroup/emin/core50_350x350/train' --dataset-name 'core50_train' --maxcount 1e6
#python -u /misc/vlgscratch5/LakeGroup/emin/webdataset-example/makeshards.py --data-dirs '/misc/vlgscratch2/LakeGroup/emin/core50_350x350/val' --dataset-name 'core50_val' --maxcount 1e6

echo "Done"
