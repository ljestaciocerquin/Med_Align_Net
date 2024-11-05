#!/bin/bash
#SBATCH --time=6-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=abd_exp                       # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                      # Partition
#SBATCH --nodelist=roentgen                      # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/liver_image_registration/Med_Align_Net/logs/abd_exp_aln1_%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch
#rsync -avv --info=progress2 --ignore-existing /data/groups/beets-tan/l.estacio/lung_data/LungCT /processing/l.estacio/

# Run your command
#python /projects/liver_image_registration/Med_Align_Net/train.py -base VXM -reg 0.1
#python /projects/liver_image_registration/Med_Align_Net/train.py -base VXM -reg 1
#python /projects/liver_image_registration/Med_Align_Net/train.py -base VXM -reg 10
#python /projects/liver_image_registration/Med_Align_Net/train.py -base VTN -reg 0.1
#python /projects/liver_image_registration/Med_Align_Net/train.py -base VTN -reg 1
#python /projects/liver_image_registration/Med_Align_Net/train.py -base VTN -reg 10
#python /projects/liver_image_registration/Med_Align_Net/train.py -base TSM -reg 0.1
#python /projects/liver_image_registration/Med_Align_Net/train.py -base TSM -reg 1
#python /projects/liver_image_registration/Med_Align_Net/train.py -base TSM -reg 10
#python /projects/liver_image_registration/Med_Align_Net/train.py -base ALN -reg 2  -name_exp 1xflow
#python /projects/liver_image_registration/Med_Align_Net/train.py -base ALN -reg 3  -name_exp 1xflow

python /projects/liver_image_registration/Med_Align_Net/train.py -base ALN -reg 1  -name_exp 1xflowxAtt
