#!/bin/bash

#SBATCH -o fabricated_reflectance-46.log
#SBATCH -n 96
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

mpirun -np 96 python -m mpi4py reproduce_fabricated_structure_figure_3.py
