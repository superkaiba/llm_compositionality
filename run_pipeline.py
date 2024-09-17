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
from dadapy import data


# os.environ['HF_HOME'] = "/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS
# os.environ['TRANSFORMERS_CACHE'] = "/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS

# CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# CHECKPOINT_STEPS += [1000 * i for i in range(1, 144)]

# CHECKPOINT_STEPS = [0, 2, 64, 512, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
# CHECKPOINT_STEPS = [1, 4, 8,16, 32, 128, 256,1000,3000,13000,53000, 103000,133000]

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
            _data = data.Data(layer_reps)
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
    # parser.add_argument("--shuffle", action="store_true", help="Shuffle the prompts")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--model_size", type=str, help="Model size")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--checkpoint_step", type=int, help="Checkpoint step")
    parser.add_argument("--n_words_correlated", type=int, nargs="+", help="Number of words correlated")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--device", type=str, help="Device", default="cuda")
    parser.add_argument("--data_dir", type=str, help="Data directory", default="/home/echeng/llm_compositionality/data")
    parser.add_argument("--results_path", type=str, help="Results path", default="/home/echeng/llm_compositionality/results_new")

    return parser.parse_args()

args = parse_args()
print(args)

model_str = f"{args.model_size}" if args.model_size == '14m' else f"{args.model_size}-deduped"

model_name = f"EleutherAI/pythia-{args.model_size}-deduped" if args.model_size not in ('14m',) else f"EleutherAI/pythia-{args.model_size}"

base_dir = f"{args.results_path}"
ensure_dir(base_dir)


df = pd.DataFrame(columns=["index", "model_name", "checkpoint_step", "n_words_correlated", "method", "layer_num", "id"])
checkpoint_step = args.checkpoint_step

# for checkpoint_step in args.checkpoint_steps:
model, tokenizer = get_model_and_tokenizer(model_name, checkpoint_step)

for n_words_correlated in args.n_words_correlated:
    RESULTS = {}

    for shuffle in (True, False):

        results_each_rs = []

        for rs in range(5):
            rep_dir = os.path.join(base_dir, model_name.replace("/", "_"), f"checkpoint_{checkpoint_step}", "representations")
            ensure_dir(rep_dir)
            rep_filename = f'{"shuffled" if shuffle else "sane"}_representations_{n_words_correlated}_words_correlated_rs{rs}.pt'
            rep_path = os.path.join(rep_dir, rep_filename)

            # we already did it.
            if os.path.exists(rep_path): 
                print('already done')
                continue

            train_data, test_data = load_prompts(n_words_correlated, args.data_dir, shuffle=shuffle, rs=rs)
            data = train_data + test_data

            if args.debug:
                data = data[:1000]
                
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

            results_each_rs.append(all_results)
        
        # Aggregate over the rs.
        agg_results = {}
        for method in ['pca', 'pr', 'twonn', 'mle']:
            all_rs = np.array([res[method] for res in results_each_rs])
            agg_results[method + '_mean'] = list(np.nanmean(all_rs, axis=0))
            agg_results[method + '_std'] = list(np.nanstd(all_rs, axis=0))

        # save as sane or shuffled
        RESULTS[mode] = agg_results

        del representations 
    
    # save results
    # Save dictionary as JSON
    save_path = f'/home/echeng/llm_compositionality/results_new/EleutherAI_pythia-{model_str}/ids_dataset_{args.dataset}_step_{args.checkpoint_step}.json'

    with open(save_path, 'w') as json_file:
        json.dump(RESULTS, json_file)

del model
del tokenizer

print(f"Results saved in directory: {base_dir}")
