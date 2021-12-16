#!/bin/bash

#SBATCH -o opt.log
#SBATCH -n 960
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 960 python -m mpi4py optimize_with_adjoints.py
