from sklearn.decomposition import PCA
import skdim
import numpy as np
import torch
from dadapy import data
import argparse
import json
from tqdm import tqdm
import pdb
import os

CHECKPOINT_FORMAT = '/home/echeng/llm_compositionality/data/saved_reps_post_finetune/prompts_{}_ckpt_{}_reps.pt' # data, step
NEW_REPS_FORMAT = '/home/echeng/llm_compositionality/results_new/EleutherAI_pythia-{}/checkpoint_{}/representations/{}_representations_{}_words_correlated_rs{}.pt'

parser = argparse.ArgumentParser(description='ID computation')

# Data selection
parser.add_argument('--model_size', type=str)
parser.add_argument('--dataset', type=int, choices=[1,2,3,4])
parser.add_argument('--step', type=int, default=None)
args = parser.parse_args()


# epoch_to_ckpt = {0:0, 0.125:25600, 0.25:51200, 1:153600, 2:307200, 3:460800, 4:614400}
# ckpt = epoch_to_ckpt[args.epoch]
# filepath = CHECKPOINT_FORMAT.format(args.dataset, ckpt)
model_str = f"{args.model_size}" if args.model_size == '14m' else f"{args.model_size}-deduped"

# Process one model, dataset combo.
RESULTS = {}

for mode in 'sane', 'shuffled':

    results_each_rs = []

    for rs in tqdm(range(5)):
        filepath = NEW_REPS_FORMAT.format(model_str, args.step, mode, args.dataset, rs)

        reps = torch.load(filepath) # list (torch tensor)
        reps = [rep.numpy().astype(float) for rep in reps]
        reps = reps[1:]
        # pdb.set_trace()

        # initialise the Data class
        all_results = {}

        # PCA
        results = {'pca': [None for _ in reps],  'pr': [None for _ in reps]}

        for layer, layer_reps in tqdm(enumerate(reps)):
            pca = PCA()
            pca.fit(layer_reps)
            explained_variances = pca.explained_variance_
            results['pca'][int(layer)] = int(np.sum(np.cumsum(pca.explained_variance_ratio_) < 0.99))
            results['pr'][layer] = sum(explained_variances)**2 / sum(explained_variances ** 2)

        all_results.update(results.copy())
        del results

        # TWONN
        results = {'twonn': [None for _ in reps], 'twonn_r': [None for _ in reps]}

        for layer, layer_reps in enumerate(reps):
            try:
                _data = data.Data(layer_reps)
                _data.remove_identical_points()

                # estimate ID
                id_twoNN, _, r = _data.compute_id_2NN()

                results['twonn'][layer] = float(id_twoNN)
                results['twonn_r'][layer] = float(r)
            except IndexError:
                continue

        all_results.update(results.copy())
        del results

        # MLE
        results = {'mle': [None for _ in reps]}
        for layer, layer_reps in enumerate(reps):
            try:
                mle = skdim.id.MLE()
                results['mle'][layer] = float(mle.fit_transform(layer_reps))
            except IndexError:
                continue

        all_results.update(results.copy())
        del results
        del reps 

        results_each_rs.append(all_results)
    
    # Aggregate over the rs.
    agg_results = {}
    for method in ['pca', 'pr', 'twonn', 'mle']:
        all_rs = np.array([res[method] for res in results_each_rs])
        agg_results[method + '_mean'] = list(np.nanmean(all_rs, axis=0))
        agg_results[method + '_std'] = list(np.nanstd(all_rs, axis=0))

    # save as sane or shuffled
    RESULTS[mode] = agg_results


# Save dictionary as JSON
save_path = f'/home/echeng/llm_compositionality/results_new/EleutherAI_pythia-{model_str}/ids_dataset_{args.dataset}_step_{args.step}.json'

with open(save_path, 'w') as json_file:
    json.dump(RESULTS, json_file)

# Clean up that checkpoint
for rs in range(5):
    for mode in 'sane', 'shuffled':
        os.remove(f'/home/echeng/llm_compositionality/results_new/EleutherAI_pythia-{model_str}/checkpoint_{args.step}/{mode}_representations_{args.dataset}_words_correlated_rs{rs}.pt')
