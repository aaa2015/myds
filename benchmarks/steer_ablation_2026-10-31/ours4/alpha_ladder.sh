#!/bin/bash
cd ~/eval || exit 1
for a in 1.0 2.0; do
  echo "########## v3_nothink alpha=$a  $(date +%H:%M:%S) ##########"
  python3 bench_safety.py --arm v3_nothink --steer-alpha "$a" \
      --output "results/safety_results_v3_a$a.jsonl" 2>&1 | grep -vE "urllib3|warnings.warn"
  echo "########## END alpha=$a  $(date +%H:%M:%S) ##########"
done
echo "LADDER DONE $(date +%H:%M:%S)"
