#!bin/bash
prefix="t"
declare variations=("_2mb" "_05mb" "t")
orders="1
2
3"
restype="M
R"
n3lo_variations="-1
0
1"

#FO results
for ord in $orders; do
    for var in ${variations[@]}; do
        if [ $ord -eq 3 ]; then
            if [ $var = "t" ]; then
                for n3lo_var in $n3lo_variations; do
                    dis_tp compute ${ord}FO ${ord}${var#"$prefix"}_${n3lo_var};
                done;
            else
                dis_tp compute ${ord}FO ${ord}${var#"$prefix"};
            fi;
        else
            dis_tp compute ${ord}FO ${ord}${var#"$prefix"};
        fi;
    done;
done;

#M and R results

for res in $restype; do
    for ord in $orders; do
        for var in ${variations[@]}; do
            if [ $ord -eq 3 ]; then
                if [ $var = "t" ]; then
                    if [ $res = "M" ]; then
                        for n3lo_var in $n3lo_variations; do
                            dis_tp compute ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"}_${n3lo_var};
                        done;
                    else
                        dis_tp compute ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"}_0;
                    fi;
                else
                    dis_tp compute ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"};
                fi;
            else
                dis_tp compute ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"};
            fi;
        done;
    done;
done;

echo "ALL DONE!"
