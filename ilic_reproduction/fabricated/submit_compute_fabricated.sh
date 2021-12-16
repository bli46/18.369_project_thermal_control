#!/bin/bash

#SBATCH -o submit_fabricated.log
#SBATCH -n 720
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 720 python -m mpi4py compute_fabricated.py
