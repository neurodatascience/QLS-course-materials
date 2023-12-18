#!/bin/bash

find ../data -type d -name 'subj*' -printf '%f\n' | sort | xargs -n 1 -I {} ./zz_mk_job.sh {}
