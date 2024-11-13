import json
import os
from datetime import datetime
import sklearn
import argparse
import numpy as np
from utils import *
# from id_measures import *
import pandas as pd
import os
import torch
from sklearn.decomposition import PCA
import skdim
from dadapy import data as dada
import pdb
import random


data_file = '/home/mbaroni/id/neurips_2024_submission/code_and_data/corpora/pile_sane_ds.txt'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def compute_pca(reps):
    results = {'pca': [None for _ in reps],  'pr': [None for _ in reps]}

    for layer, layer_reps in tqdm(enumerate(reps)):
        pca = PCA()
        pca.fit(layer_reps)
        explained_variances = pca.explained_variance_
        results['pca'][int(layer)] = int(np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.99))
        results['pr'][layer] = sum(explained_variances)**2 / sum(explained_variances ** 2)

    return results

def compute_twonn(reps):
    results = {'twonn': [None for _ in reps], 'twonn_r': [None for _ in reps]}

    for layer, layer_reps in enumerate(reps):
        try:
            _data = dada.Data(layer_reps)
            _data.remove_identical_points()

            # estimate ID
            id_twoNN, _, r = _data.compute_id_2NN()

            results['twonn'][layer] = float(id_twoNN)
            results['twonn_r'][layer] = float(r)
        except IndexError:
            continue
    return results

def compute_mle(reps):
    results = {'mle': [None for _ in reps]}
    for layer, layer_reps in enumerate(reps):
        try:
            mle = skdim.id.MLE()
            results['mle'][layer] = float(mle.fit_transform(layer_reps))
        except IndexError:
            continue
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline also with prompt shuffling.")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--model", type=str, help="Model size. llama, mistral, 14m, 70m etc for pythia")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--checkpoint_step", type=int, help="Checkpoint step", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--device", type=str, help="Device", default="cuda")
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument("--data_dir", type=str, help="Data directory", default="/home/echeng/llm_compositionality/data")
    parser.add_argument("--results_path", type=str, help="Results path", default="/home/echeng/llm_compositionality/results_new")

    return parser.parse_args()

args = parse_args()
print(args)

if 'llama' not in args.model and 'mistral' not in args.model:
    model_str = f"{args.model}" if args.model == '14m' else f"{args.model}-deduped"
    model_name = f"EleutherAI/pythia-{args.model}-deduped" if args.model not in ('14m',) else f"EleutherAI/pythia-{args.model}"
elif 'llama' == args.model:
    model_name = 'meta-llama/Meta-Llama-3-8B'
elif 'mistral' == args.model:
    model_name = 'mistralai/Mistral-7B-v0.1'

base_dir = f"{args.results_path}"
ensure_dir(base_dir)

checkpoint_step = args.checkpoint_step

model, tokenizer = get_model_and_tokenizer(model_name, checkpoint_step)

RESULTS = {}

results_each_rs = []


# Load prompts 
all_data = []
f = open(data_file)
for line in f:
    input_fields = line.strip("\n").split("\t")
    snippet = input_fields[0].split(" ")[:17]
    if args.shuffle: random.shuffle(snippet)
    all_data.append(" ".join(snippet))
f.close()

for rs in tqdm(range(5)):
    data = all_data[rs * 10000: (1+rs) * 10000]
        
    with torch.no_grad():
        representations = get_reps_from_llm(model, tokenizer, data, args.device, args.batch_size)

    reps = [rep.numpy().astype(float) for rep in representations][1:]

    # Compute the ID
    all_results = {}

    # PCA
    results = compute_pca(reps)
    all_results.update(results.copy())
    del results

    # TWONN
    results = compute_twonn(reps)
    all_results.update(results.copy())
    del results

    # MLE
    results = compute_mle(reps)
    all_results.update(results.copy())

    del results
    del reps 
    del representations 

    results_each_rs.append(all_results)

# Aggregate over the rs.
for method in ['pca', 'pr', 'twonn', 'mle']:
    all_rs = np.array([res[method] for res in results_each_rs])
    RESULTS[method + '_mean'] = list(np.nanmean(all_rs, axis=0))
    RESULTS[method + '_std'] = list(np.nanstd(all_rs, axis=0))


# save results
# Save dictionary as JSON
if 'llama' in args.model or 'mistral' in args.model:
    save_path = f'/home/echeng/llm_compositionality/results_new/{args.model}/ids_dataset_pile{"" if not args.shuffle else "_shuffled"}.json'
else:
    save_path = f'/home/echeng/llm_compositionality/results_new/EleutherAI_pythia-{model_str}/ids_dataset_pile_step_{args.checkpoint_step}{"" if not args.shuffle else "_shuffled"}.json'

with open(save_path, 'w') as json_file:
    json.dump(RESULTS, json_file)

del model
del tokenizer

print(f"Results saved in directory: {base_dir}")
