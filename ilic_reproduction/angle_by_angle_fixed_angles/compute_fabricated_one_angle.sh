#!/bin/bash

#SBATCH -o fabricated_reflectance-angle_by_angle.log
#SBATCH -n 192
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 192 python -m mpi4py fabricated_structure_one_angle.py
