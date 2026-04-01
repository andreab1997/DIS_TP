#!bin/bash

plot_path="project/plots"
heavy_quark="5"

for ord in NLO NNLO N3LO; do
    for ob in F2 FL; do
        dis_tp plot $plot_path $ob $ord $heavy_quark plot_single_obs_ord;
        dis_tp plot $plot_path $ob $ord $heavy_quark plot_single_obs_noband_ord;
    done;
done;

# Order Comparison

for ob in F2 FL; do
    dis_tp plot $plot_path $ob NLO $heavy_quark plot_fonll_order_comparison;
done;

for ob in F2 FL; do
    dis_tp plot $plot_path $ob NLO $heavy_quark plot_massive_order_comparison;
done;

echo "ALL DONE!"
