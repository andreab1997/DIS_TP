
CORES=1

for nf in 4 5; do
  for var in 0 1 -1; do
    for kind in L 2; do
      for chan in q g; do
        if [ "$kind" = 'L' ]; then
          echo "Computing "$kind$chan" and saving in CL"$chan"_3_til"
        else
          echo "Computing "$kind$chan" and saving in C"$chan"_3_til"
        fi
        dis_tp grids tilde -n $CORES -v $var $nf $kind$chan 3
      done
    done
  done
done
