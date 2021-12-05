#!/bin/bash
#SBATCH -n 20
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp
np=20;

a=4.3;
r=1.72;

for t in `seq 0 5 30`; do
    mpirun -np ${np} python3 -u emitter.py -empty -aa ${a} -theta ${t} > flux0_a${a}_theta${t}.out;
    grep flux1: flux0_a${a}_theta${t}.out |cut -d , -f2- > flux0_a${a}_theta${t}.dat;

    mpirun -np ${np} python3 -u emitter.py -aa ${a} -rr ${r} -theta ${t} > flux_a${a}_r${r}_theta${t}.out;
    grep flux1: flux_a${a}_r${r}_theta${t}.out |cut -d , -f2- > flux_a${a}_r${r}_theta${t}.dat;
done;