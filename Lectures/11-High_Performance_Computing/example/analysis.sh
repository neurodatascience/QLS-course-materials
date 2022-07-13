#!/bin/bash

## pull the subject ID argument
SUBJ=$1

## wait 2.5 minutes
sleep 150

## report back about the node the job ran on
HN=`hostname`

## catch the date
DT=`date`

## print the current working directory
WD=`pwd`

cat << EOF > ../results/output_${SUBJ}.txt

Running subject: ${SUBJ}
    on host: ${HN}

Current Working Directory: ${WD}

Waiting 2.5 minutes...

Job finished at: ${DT}

EOF
