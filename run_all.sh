#!/bin/bash
cd "$(dirname "$0")"
N=${N:-800}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p logs/exp
CFGS=(
"full|"
"flat|tp=(1,0,0,0)"
"nofric|fmin=1.0 fmax=1.0"
"nopush|pv=0.0"
"nodelay|pdl=0.0"
"nonoise|nav=0.0 ngp=0.0 ndq=0.0 ndv=0.0"
)
for s in $SEEDS;do
  for c in "${CFGS[@]}";do
    name=${c%%|*}
    sets=${c#*|}
    last=logs/exp/${name}_s${s}/model_$((N-1)).pt
    if [ -f "$last" ];then echo "skip $name s$s";continue;fi
    echo "=== $name seed $s $(date)"
    python3 train.py --name $name --seed $s --iters $N ${sets:+--set $sets} >> logs/exp/${name}_s${s}.out 2>&1
  done
done
echo "all done $(date)"
