#!bin/bash
prefix="t"
declare variations=("_2mb" "_05mb" "t")
restype="M R"

#FO results
for ord in 1 2 3; do
    for var in ${variations[@]}; do
        if [ $ord -eq 3 ]; then
            if [ $var = "t" ]; then
                for n3lo_var in -1 0 1; do
                    dis_tp compute -n 2 ${ord}FO ${ord}${var#"$prefix"}_${n3lo_var};
                done;
            else
                dis_tp compute -n 2 ${ord}FO ${ord}${var#"$prefix"};
            fi;
        else
            dis_tp compute -n 2 ${ord}FO ${ord}${var#"$prefix"};
        fi;
    done;
done;

#M and R results

for res in "M" "R"; do
    for ord in 1 2 3; do
        for var in ${variations[@]}; do
            if [ $ord -eq 3 ]; then
                if [ $var = "t" ]; then
                    if [ $res = "M" ]; then
                        for n3lo_var in -1 0 1; do
                            dis_tp compute -n 2 ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"}_${n3lo_var};
                        done;
                    else
                        dis_tp compute -n 2 ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"}_0;
                    fi;
                else
                    dis_tp compute -n 2 ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"};
                fi;
            else
                dis_tp compute -n 2 ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"};
            fi;
        done;
    done;
done;

echo "ALL DONE!"
