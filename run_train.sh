#!/bin/bash
#SBATCH --time=3-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=abd                    # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                        # Partition
#SBATCH --nodelist=roentgen                    # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/disentanglement_methods/Med_Align_Net/logs/aln_abd_20_3T_01_nr%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch
#rsync -avv --info=progress2 --ignore-existing /data/groups/beets-tan/l.estacio/lung_data/LungCT /processing/l.estacio/

# Run your command
#python /projects/disentanglement_methods/Med_Align_Net/train.py
#python /projects/disentanglement_methods/Med_Align_Net/train.py -base VTN
#python /projects/disentanglement_methods/Med_Align_Net/train.py -base TSM
python /projects/disentanglement_methods/Med_Align_Net/train.py -base ALN
#python /projects/disentanglement_methods/Med_Align_Net/train.py -base ALN -r 20 -d /processing/l.estacio/LungCT/LungCT_dataset.json -rd /processing/l.estacio/LungCT/ 