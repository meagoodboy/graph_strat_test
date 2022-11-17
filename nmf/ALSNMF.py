import sys
import pandas as pd
import numpy as np
from libnmf.alsnmf import ALSNMF
import seaborn as sns
import gower
from data_procesing import process_data

encoding = str(sys.argv[1])
clinical_file = str(sys.argv[2])
rank = int(str(sys.argv[3]))
max_iter = int(str(sys.argv[4]))


save_name = encoding[:-4]
print("save_file : ",save_name)

enc = pd.read_csv(encoding, index_col=0).fillna(0)
cols = enc.columns
for col in cols:
    enc[col] = enc[col].astype(float)

enc_np = np.array(enc)
enc_np.shape
enc_np = gower.gower_matrix(enc_np)
print(enc_np.shape)

alsnmf= ALSNMF(enc_np, rank=rank)
alsnmf.compute_factors(max_iter= max_iter)

W  = pd.DataFrame(alsnmf.W, index=enc.index)

clinical = pd.read_csv(clinical_file, index_col=0)

data, features = process_data(W, clinical, "ajcc_pathologic_stage")
data.columns = ["x","y"]

plot_emb = data.copy()
plot_emb['label'] = features
plot = sns.scatterplot(data=plot_emb, x='x', y='y', hue='label',s=70, alpha=0.8, palette="coolwarm")
plot.set(title="ALSNMF")
fig = plot.get_figure()
fig.savefig(save_name+ "_alsnmf_scatter.png")

data.to_csv(save_name+"_alsnmf.csv")