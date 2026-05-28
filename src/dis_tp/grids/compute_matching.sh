#!bin/bash

CORES=1

for nf in 4 5; do
  for fl in bq bg; do
    echo "Computing "$fl" and saving in M"$fl"_3"
    dis_tp grids matching -n $CORES $nf $fl 3
  done
done
