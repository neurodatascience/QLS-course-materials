#!/bin/bash

SUBJ=$1

cat << EOF > ./jobs/demo_${SUBJ}.slurm
#!/bin/bash

#SBATCH --job-name=demo_${SUBJ}                  # job name
#SBATCH --nodes=1                               # run on a single node
#SBATCH --ntasks=1                              # run on a single CPU
#SBATCH --cpus-per-task=1                       # run on a single core
#SBATCH --mem=1gb                               # job memory request
#SBATCH --time=00:05:00                         # time limit hrs:min:sec
#SBATCH --error=./jobs/logs/demo_${SUBJ}_%j.err  # standard error from job
#SBATCH --output=./jobs/logs/demo_${SUBJ}_%j.out # standard output from job
#SBATCH --account=def-jbpoline                  # define your affiliation

## ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
## MODIFY THE RESOURCE REQUSTS ABOVE FOR YOUR JOB

## MODIFY THE CODE BELOW TO CALL YOUR ANALYSIS
## vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

## your job script generalized to a subject ID
bash ./zz_analysis.sh ${SUBJ}

##
## use a container
##

#module load apptainer

## your job script generalized to a subject ID
#apptainer exec -H /scratch/$USER/hpc-demo -B /scratch/$USER/hpc-demo ../container/ubuntu.sif bash ./zz_analysis.sh ${SUBJ}

EOF
