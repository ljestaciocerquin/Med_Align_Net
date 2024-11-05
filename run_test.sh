#!/bin/bash
#SBATCH --time=4-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=test_elastix                    # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                        # Partition
#SBATCH --nodelist=roentgen                    # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/liver_image_registration/Med_Align_Net/logs/test_%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch
#rsync -avv --info=progress2 --ignore-existing /data/groups/beets-tan/l.estacio/lung_data/LungCT /processing/l.estacio/

# Run your command
#Abdomen

#VXM
###python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VXM/train/Sep16-203808_Abtrain_VXMx1___reg0.1/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VXM/train/Sep17-110518_Abtrain_VXMx1___reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ #-sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VXM/train/Sep18-014522_Abtrain_VXMx1___reg10.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ #-sn

#VTN
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VTN/train/Sep16-204228_Abtrain_VTNx3___reg0.1/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ #-sn
###python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VTN/train/Sep17-122400_Abtrain_VTNx3___reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VTN/train/Sep18-042341_Abtrain_VTNx3___reg10.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ #-sn

# TSM
###python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/TSM/train/Sep18-214410_Abtrain_TSMx1___reg0.1/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/TSM/train/Sep19-141338_Abtrain_TSMx1___reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ #-sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/TSM/train/Sep20-064524_Abtrain_TSMx1___reg10.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn

# ALN
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep18-215311_Abtrain_ALNx1_1xflow__reg0.1/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep19-131330_Abtrain_ALNx1_1xflow__reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep20-051025_Abtrain_ALNx1_1xflow__reg10.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn

#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep22-172410_Abtrain_ALNx1_20xflow__reg0.1/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep23-151805_Abtrain_ALNx1_20xflow__reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep24-132646_Abtrain_ALNx1_20xflow__reg10.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn

# ALN Attention
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Oct07-182843_Abtrain_ALNx1_1xflowxAtt__reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn

# Dual Encoder affine and elastic:
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep30-194807_Abtrain_ALNx1_1xflowxAvpAff__reg1.0/model_wts/epoch_100.pth  -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ #-sn

#
python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -ua -en ants_Abd -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VXM/train/Sep16-203808_Abtrain_VXMx1___reg0.1/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn
python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -ue -en elastix_Abd -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/VXM/train/Sep16-203808_Abtrain_VXMx1___reg0.1/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ -sn



#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep28-061053_Abtrain_ALNx1_1xflow__reg2.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ 
#python /projects/liver_image_registration/Med_Align_Net/eval/eval.py -c /projects/liver_image_registration/Med_Align_Net/logs/Abdomen/ALN/train/Sep29-052304_Abtrain_ALNx1_1xflow__reg3.0/model_wts/epoch_100.pth -d /processing/l.estacio/AbdomenCTCT/AbdomenCTCT_dataset.json -rdir /processing/l.estacio/AbdomenCTCT/ 
