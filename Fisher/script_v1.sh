#!/bin/sh
#SBATCH -J SPARTA
#SBATCH --partition=batch4
#SBATCH --get-user-env
#SBATCH -o ./job.%j.out
#SBATCH -e ./job.%j.err
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 23:59:00

export OMP_NUM_THREADS=8

srun python3.8 ./fisher_sn_quijote.py