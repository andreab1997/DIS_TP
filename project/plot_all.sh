#!bin/bash
orders="NLO
NNLO
N3LO
"

obs="F2
FL
"

plot_path="project/plots"
heavy_quark="5"

for ord in $orders; do
    for ob in $obs; do
        dis_tp plot $plot_path $ob $ord $heavy_quark;
    done;
done;

echo "ALL DONE!"
