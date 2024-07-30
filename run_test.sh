#!/bin/bash
#SBATCH --time=7-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=test                    # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                        # Partition
#SBATCH --nodelist=roentgen                    # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/disentanglement_methods/Med_Align_Net/logs/test_%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch
rsync -avv --info=progress2 --ignore-existing /data/groups/beets-tan/l.estacio/lung_data/LungCT /processing/l.estacio/

# Run your command
#python /projects/disentanglement_methods/Med_Align_Net/eval/eval.py -c /projects/disentanglement_methods/Med_Align_Net/logs/lung/VXM/train/Jun24-204642_lutrain_VXMx1___/model_wts/epoch_100.pth -sn
#python /projects/disentanglement_methods/Med_Align_Net/eval/eval.py -c /projects/disentanglement_methods/Med_Align_Net/logs/lung/VTN/train/Jul10-004547_lutrain_VTNx3___/model_wts/epoch_100.pth -sn
#python /projects/disentanglement_methods/Med_Align_Net/eval/eval.py -c /projects/disentanglement_methods/Med_Align_Net/logs/lung/TSM/train/Jul10-164631_lutrain_TSMx1___/model_wts/epoch_100.pth -sn
#python /projects/disentanglement_methods/Med_Align_Net/eval/eval.py -ue -en elastix -sn
#python /projects/disentanglement_methods/Med_Align_Net/eval/eval.py -ua -en ants -sn
python /projects/disentanglement_methods/Med_Align_Net/eval/eval.py -sn
