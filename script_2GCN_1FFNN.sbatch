#!/bin/bash
#SBATCH --time=100:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=GNN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=antonio.tudisco@polito.it
#SBATCH --partition=cuda
#SBATCH --output=job_name_%j.log

# Load CUDA and Python
module load nvidia/cudasdk/10.1
#module load intel/python/3

# Ensure Conda is initialized properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn_py10
conda env list
conda info --envs
conda list scikit-learn


# Debugging Information
echo "Running on host: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Python version: $(python --version)"
echo "Conda environment: $(which python)"
echo "Installed Python packages:"
pip list | grep torch
pip list | grep scikit-learn
# Run Python script
python gnn_GAT_GCNResNet2_FFNN1Layer.py
