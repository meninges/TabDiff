import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
import argparse
import json

parser = argparse.ArgumentParser(description='DCR evaluation of imputed data')
parser.add_argument('--dataname', type=str, default=None, help='Name of dataset.')
args = parser.parse_args()

dataname = args.dataname

if dataname is None:
    raise ValueError("No dataname given, please provide the name of your dataset")
elif not dataname.endswith("_dcr"):
    raise ValueError("Dataset given is not a DCR set, please retrain with 50/50 split or rename appropriately!")

info_path = f'data/{dataname}/info.json'
with open(info_path, 'r') as f:
    info = json.load(f)

real_data = pd.read_csv(f"synthetic/{dataname}/real.csv")
test_data = pd.read_csv(f"synthetic/{dataname}/test.csv")
syn_data = pd.read_csv(f"impute/{dataname}/learnable_schedule/0.csv")

if info["task_type"] == "regression":
    num_col_idx = info["target_col_idx"]
    cat_col_idx = []
else:
    num_col_idx = []
    cat_col_idx = info["target_col_idx"]

num_ranges = []

real_data.columns = list(np.arange(len(real_data.columns)))
syn_data.columns = list(np.arange(len(real_data.columns)))
test_data.columns = list(np.arange(len(real_data.columns)))
for i in num_col_idx:
    num_ranges.append(real_data[i].max() - real_data[i].min()) 

num_ranges = np.array(num_ranges)


num_real_data = real_data[num_col_idx]
cat_real_data = real_data[cat_col_idx]
num_syn_data = syn_data[num_col_idx]
cat_syn_data = syn_data[cat_col_idx]
num_test_data = test_data[num_col_idx]
cat_test_data = test_data[cat_col_idx]

num_real_data_np = num_real_data.to_numpy()
cat_real_data_np = cat_real_data.to_numpy().astype('str')
num_syn_data_np = num_syn_data.to_numpy()
cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
num_test_data_np = num_test_data.to_numpy()
cat_test_data_np = cat_test_data.to_numpy().astype('str')

if info["task_type"] != "regression":
    encoder = OneHotEncoder()
    cat_complete_data_np = np.concatenate([cat_real_data_np, cat_test_data_np], axis=0)
    encoder.fit(cat_complete_data_np)
    # encoder.fit(cat_real_data_np)
    cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
    cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
    cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()
else:
    cat_real_data_oh = np.empty((num_real_data_np.shape[0], 0))
    cat_syn_data_oh = np.empty((num_syn_data_np.shape[0], 0))
    cat_test_data_oh = np.empty((num_test_data_np.shape[0], 0))

num_real_data_np = num_real_data_np / num_ranges
num_syn_data_np = num_syn_data_np / num_ranges
num_test_data_np = num_test_data_np / num_ranges

real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

device = torch.device("cuda")

real_data_th = torch.tensor(real_data_np).to(device)
syn_data_th = torch.tensor(syn_data_np).to(device)  
test_data_th = torch.tensor(test_data_np).to(device)

dcrs_real = []
dcrs_test = []
batch_size = 10000 // max(cat_real_data_oh.shape[1], 1)   # This esitmation should make sure that dcr_real and dcr_test can be fit into 10GB GPU memory

for i in range((syn_data_th.shape[0] // batch_size) + 1):
    if i != (syn_data_th.shape[0] // batch_size):
        batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
    else:
        batch_syn_data_th = syn_data_th[i*batch_size:]
        
    dcr_real = (batch_syn_data_th[:, None] - real_data_th).abs().sum(dim = 2).min(dim = 1).values
    dcr_test = (batch_syn_data_th[:, None] - test_data_th).abs().sum(dim = 2).min(dim = 1).values
    dcrs_real.append(dcr_real)
    dcrs_test.append(dcr_test)
    
dcrs_real = torch.cat(dcrs_real)
dcrs_test = torch.cat(dcrs_test)

score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]


print(score)

