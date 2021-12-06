#!/bin/bash

#SBATCH -o back_fabricated_reflectance-angle_by_angle_doubled_pml_and_res.log
#SBATCH -n 192
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 192 python -m mpi4py fabricated_structure_back_norm.py
