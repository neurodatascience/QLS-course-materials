#!/bin/bash

## create the jobs, logs, and output folders.
mkdir jobs
mkdir jobs/logs
mkdir results

## for every subject, create a job file
for subj in subj01 subj02 subj03 subj04 subj05; do

    cat << EOF > ./jobs/demo_${subj}.slurm
#!/bin/bash

#SBATCH --job-name=demo_${subj}                  # job name
#SBATCH --nodes=1                               # run on a single node
#SBATCH --ntasks=1                              # run on a single CPU
#SBATCH --cpus-per-task=1                       # run on a single core
#SBATCH --mem=1gb                               # job memory request
#SBATCH --time=00:05:00                         # time limit hrs:min:sec
#SBATCH --error=./jobs/logs/demo_${subj}_%j.err  # standard error from job
#SBATCH --output=./jobs/logs/demo_${subj}_%j.out # standard output from job

##
## MODIFY THE RESOURCE REQUESTS ABOVE FOR YOUR JOB
##
## MODIFY THE CODE BELOW TO CALL YOUR ANALYSIS
##

## your job script generalized to a subject ID
bash ./analysis.sh ${subj}

EOF

done
