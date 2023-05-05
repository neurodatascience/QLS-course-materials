#!/bin/bash

## pull the subject ID argument
SUBJ=$1

## wait 2.5 minutes
sleep 150

## point to the file
DATA=`cat ../data/${SUBJ}/${SUBJ}-data.txt`

## report back about the node the job ran on
HN=`hostname`

## catch the date
DT=`date`

## print the current working directory
WD=`pwd`

## what to look for in a container
if [ -e /.singularity.d ];
then
  AP="I am running in a container."
else
  AP="I am not running in a container."
fi

cat << EOF > ../results/output_${SUBJ}.txt

Running subject: ${SUBJ} on host: ${HN}

Current Working Directory: ${WD}

${SUBJ} Data: 
${DATA}

${AP}

Waiting 2.5 minutes...

Job finished at: ${DT}

EOF
