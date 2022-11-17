matrix=$1
clin="../../kirp_clin.csv"
max_iter=100

python3 NMF.py $matrix $clin 2 $max_iter
python3 GNMF.py $matrix $clin 2 $max_iter
python3 PNMF.py $matrix $clin 2 $max_iter
python3 FPDNMF.py $matrix $clin 2 $max_iter
python3 ALSNMF.py $matrix $clin 2 $max_iter
