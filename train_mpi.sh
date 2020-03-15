#!/bin/bash

# Slurm sbatch options
#SBATCH -o mpi/mpi_%j.log
#SBATCH -n 17
#SBATCH -p gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --time=10-00:00:00
#SBATCH -N 1

# Initialize the module command first
source /etc/profile

# Load MPI module
module load mpi/openmpi-4.0

# Load Anaconda module
module load anaconda/2020a

# Call your script as you would from the command line
mpirun python -B train.py --env 5_vs_5 --model maddpg --steps 2000000 --reward_shape
