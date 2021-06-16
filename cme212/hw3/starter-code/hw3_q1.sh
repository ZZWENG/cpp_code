#!/bin/bash
#SBATCH -o slurm_%j.out
#SBATCH -p CME
#SBATCH --gres gpu:1


### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------

echo "Starting at `date`"
echo
make

echo
echo Output from main_q1
echo ----------------
./main_q1