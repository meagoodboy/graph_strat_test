x=$(find ../data/gae -name "*.csv")
for file in $x
do
    bash runall.sh $file
done