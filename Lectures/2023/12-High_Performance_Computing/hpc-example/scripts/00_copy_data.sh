#!/bin/bash

echo "Making job, data, and output directories..."

## create the jobs, logs, and output folders.
mkdir -p jobs/logs
mkdir ../results
mkdir ../container

## create the "data" to analyze
mkdir -p ../data/{subj01,subj02,subj03,subj04,subj05,subj06}

echo "Name: Graham Chapman" > ../data/subj01/subj01-data.txt
echo "Name: John Cleese" > ../data/subj02/subj02-data.txt
echo "Name: Terry Gilliam" > ../data/subj03/subj03-data.txt
echo "Name: Eric Idle" > ../data/subj04/subj04-data.txt
echo "Name: Terry Jones" > ../data/subj05/subj05-data.txt
echo "Name: Michael Palin" > ../data/subj06/subj06-data.txt

echo "Building demo container..."

## build a simple container
module load singularity
singularity build ../container/ubuntu.sif docker://ubuntu:latest
