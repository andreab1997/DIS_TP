#!bin/bash
prefix="t"
declare variations=("_2mb" "_05mb" "t")
orders="1
2
3"
restype="M
R"

#FO results
for ord in $orders; do
    for var in ${variations[@]}; do
        dis_tp compute ${ord}FO ${ord}${var#"$prefix"};
    done;
done;

#M and R results

for res in $restype; do
    for ord in $orders; do
        for var in ${variations[@]}; do
            dis_tp compute ${ord}${res}${var#"$prefix"} ${ord}${var#"$prefix"};
        done;
    done;
done;

echo "ALL DONE!"
