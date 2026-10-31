#!/bin/bash
cd ~/eval || exit 1
for arm in base_nothink base_think v2_nothink v3_nothink; do
  echo "########## ARM $arm  $(date +%H:%M:%S) ##########"
  python3 bench_safety.py --arm "$arm" 2>&1
  echo "########## END $arm  $(date +%H:%M:%S) ##########"
done
echo "ALL DONE $(date +%H:%M:%S)"
