from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import skdim
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pickle
import numpy as np
import torch
from dadapy import data
import argparse
import json
import pdb

CHECKPOINT_FORMAT = '/home/echeng/llm_compositionality/data/saved_reps_post_finetune/prompts_{}_ckpt_{}_reps.pt' # data, step

parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--dataset', type=int, choices=[1, 2, 3, 4])
parser.add_argument('--method', type=str, default='gride')
parser.add_argument('--epoch', type=int, choices=[0, 1, 2, 3, 4])
parser.add_argument('--random_seed', type=int, default=32)
parser.add_argument('--step', type=int, default=None)
args = parser.parse_args()

np.random.seed(args.random_seed)

epoch_to_ckpt = [0, 153600, 307200, 460800, 614400]
ckpt = epoch_to_ckpt[args.epoch]
filepath = CHECKPOINT_FORMAT.format(args.dataset, ckpt)

reps = torch.load(filepath) # list (torch tensor)
reps = [rep.numpy().astype(float) for rep in reps]
reps = reps[1:]
# pdb.set_trace()

# initialise the Data class
all_results = {}

# PCA
results = {'pca_id': [None for _ in reps],  'pr_id': [None for _ in reps], 'explained_var': [None for _ in reps], 'eigenspectrum': [None for _ in reps]}

for layer, layer_reps in enumerate(reps):
    pca = PCA()
    pca.fit(layer_reps)
    explained_variances = pca.explained_variance_
    results['pca_id'][int(layer)] = int(np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.99))
    results['pr_id'][layer] = sum(explained_variances)**2 / sum(explained_variances ** 2)
    results['eigenspectrum'][int(layer)] = list(explained_variances)
    results['explained_var'][int(layer)] = 0.99

all_results['pca'] = results 

# TWONN
results = {'id': [None for _ in reps], 'r': [None for _ in reps], 'err': [None for _ in reps]}

for layer, layer_reps in enumerate(reps):
    try:
        _data = data.Data(layer_reps)
        _data.remove_identical_points()

        # estimate ID
        id_twoNN, _, r = _data.compute_id_2NN()
        print('Estimated twoNN with r=typical distance between point i and its neighbor')
        print(id_twoNN)
        print(r)
        results['id'][layer] = float(id_twoNN)
        results['r'][layer] = float(r)
    except IndexError:
        continue

all_results['twonn'] = results 

# MLE
results = {'id': [None for _ in reps]}
for layer, layer_reps in enumerate(reps):
    try:
        mle = skdim.id.MLE()
        results['id'][layer] = float(mle.fit_transform(layer_reps))
    except IndexError:
        continue

all_results['mle'] = results 

# GRIDE
results = {layer: {'id': [],
                    'err': [],
                    'r': []
                    } for layer in range(1, len(reps) + 1)}
for layer, layer_reps in enumerate(reps):
    _data = data.Data(layer_reps)
    _data.remove_identical_points()

    # estimate ID
    ids_scaling, ids_scaling_err, rs_scaling = _data.return_id_scaling_gride(range_max = 2**13)
    results[layer + 1]['r'] = rs_scaling.tolist()
    results[layer + 1]['err'] = ids_scaling_err.tolist()
    results[layer + 1]['id'] = ids_scaling.tolist()

all_results['gride'] = results

# Save dictionary as JSON
save_path = f'/home/echeng/llm_compositionality/data/saved_reps_post_finetune/prompts_{args.dataset}_step_{ckpt}_{args.method}.json'

with open(save_path, 'w') as json_file:
    json.dump(all_results, json_file)
