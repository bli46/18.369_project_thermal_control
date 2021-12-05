#!/bin/bash

#SBATCH -o script1.sh.log-%j

#SBATCH -n 20
#SBATCH --exclusive

source /etc/profile
module load anaconda/2021a
source activate pmp

np=20;

for a in `seq 4.1 0.1 5.0`; do
    mpirun -np ${np} python3 -u emitter.py -empty -aa {a} > flux0_a${a}.out;
    grep flux1: flux0_a${a}.out |cut -d , -f2- > flux0_a${a}.dat;
    for r_frac in `seq 0.1 0.1 0.4`; do
        r=$(printf "%0.2f" `echo "${a}*${r_frac}" |bc`);           
        mpirun -np ${np} python3 -u emitter.py -aa ${a} -rr ${r} > flux_a${a}_r${r}.out;
        grep flux1: flux_a${a}_r${r}.out |cut -d , -f2- > flux_a${a}_r${r}.dat;
    done;
done;