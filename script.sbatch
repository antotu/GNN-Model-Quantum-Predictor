#!/bin/bash
#SBATCH --time=100:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=python-ex
#SBATCH --mail-type=ALL
#SBATCH --partition=cuda


#SBATCH --job-name=GNN
#SBATCH --mail-user=antonio.tudisco@polito.it

#SBATCH --output=job_name_%j.log
module load nvidia/cudasdk/10.1
module load intel/python/3
source /home/atudisco/miniconda3/bin/activate
conda activate gnn

# Copia dei file da storage HOME a storage locale del nodo
#conda activate gnn
python gnn.py