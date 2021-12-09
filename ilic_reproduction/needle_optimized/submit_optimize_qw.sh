#!/bin/bash

#SBATCH -o optimize_qw.log
#SBATCH -n 700
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 700 python -m mpi4py optimize_qw.py
