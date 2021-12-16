#!/bin/bash

#SBATCH -o test.log
#SBATCH -n 180
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 180 python -m mpi4py test_some_gradients.py
