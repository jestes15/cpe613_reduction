#!/usr/bin/env bash
module load gcc/11.2.1

# delete previous output from slurm
rm -rf *.out

# submit the job to the queue
sbatch submission.slurm > submission.txt

if [[ ! `cat submission.txt` =~ "Submitted" ]]; then
   echo "Issue submitting..."
   cat submission.txt
   rm -f submission.txt
   exit 1
fi

JOBNUM=`cat submission.txt | awk '{print $4}'`

rm -f submission.txt[