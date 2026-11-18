#!/bin/bash
cd ~/hexa_mj
while pgrep -f run_all.sh >/dev/null; do sleep 30; done
n=$(ls logs/exp/*/model_799.pt | wc -l)
echo "finished runs: $n"
[ "$n" -eq 18 ] || { echo "expected 18, not evaluating"; exit 1; }
python3 eval.py logs/exp/*_s?/ -o results/eval.csv && python3 aggregate.py results/eval.csv -o results
echo "post done $(date)"
