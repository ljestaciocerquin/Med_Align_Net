#!/bin/bash
#SBATCH --time=1-00:00:00                        # Time limit hrs:min:sec
#SBATCH --job-name=lung_vxm                    # Job name
#SBATCH --qos=a6000_qos
#SBATCH --partition=rtx8000                        # Partition
#SBATCH --nodelist=roentgen                    # Node name
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=30G
#SBATCH --output=/projects/disentanglement_methods/Med_Align_Net/logs/vxm%j.log   # Standard output and error log
pwd; hostname; date

# Activate conda environment pyenv
source /home/l.estacio/miniconda3/bin/activate pytorch

# Run your command
python /projects/disentanglement_methods/Med_Align_Net/train.py