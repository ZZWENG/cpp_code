#!/bin/bash

#SBATCH --time=00:40:00
#SBATCH --partition=CME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --job-name=fp-profile
#SBATCH --output=fp-%j.out
#SBATCH --error=fp-%j.err

### ---------------------------------------
### BEGINNING OF EXECUTION
### ---------------------------------------
echo The master node of this job is `hostname`
echo This job runs on the following nodes:
echo `scontrol show hostname $SLURM_JOB_NODELIST`
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `echo $SLURM_SUBMIT_DIR`"
echo
echo Output from code
echo ----------------
### end of information preamble

cd $SLURM_SUBMIT_DIR

# Choose your option

# Just run the code with some default parameters
#nsys profile -o nsys mpirun -np 4 ./main -e 1

#ncu --set detailed -o ncompute_detailed ./main -e 1
#ncu -o ncompute ./main -e 1
#mpirun -np 4 ./main -e 1 -b 50
#mpirun -np 4 ./main -e 1 -b 100
#mpirun -np 4 ./main -e 1 -b 200
#mpirun -np 4 ./main -e 1 -b 400
#mpirun -np 4 ./main -e 1 -b 600
#mpirun -np 4 ./main -e 1 -b 800

# Grading mode 4
#echo "./main -g 1"
#./main -g 1

# Comment out to run the rest
exit 0

# Loop through grading modes 1, 2, and 3
# Errors are outputted in the directories Outputs_{1,2,3}
for mode in 1 2 3 4
do
    echo "mpirun -np 4 ./main -g ${mode}"
    time mpirun -np 4 ./main -g ${mode}
    if [ -d Outputs_${mode} ]
    then
        rm -r Outputs_${mode}
    fi 
    mkdir Outputs_${mode}
    cp Outputs/{CpuGpuDiff.txt,NNErrors.txt} Outputs_${mode}/
done
