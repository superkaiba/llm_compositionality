import json
import os
from datetime import datetime
import skdim
import sklearn
from dadapy import Data
import argparse
import numpy as np
from utils import *
import pandas as pd

# MODEL_SIZES = ["70m", "160m", "410m" , "1b", "1.4b", "2.8b", "6.9b", "12b"]
MODEL_SIZES = ["1b", "1.4b", "2.8b", "6.9b", "12b"]
# MODEL_SIZES = ["12b"]
MODEL_NAMES = [f"EleutherAI/pythia-{model_size}-deduped" for model_size in MODEL_SIZES]
# CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# CHECKPOINT_STEPS += [1000 * i for i in range(1, 144)]

CHECKPOINT_STEPS = [16, 512, 4000, 16000, 32000, 64000, 100000, 143000]
N_WORDS_CORRELATED_LIST = [i for i in range(1, 5)]
DEVICE = 'cuda'
BATCH_SIZE = 64
N_REPS = 1000
METHODS = {
  "mle": skdim.id.MLE(),
  # "pca": skdim.id.lPCA()
} 
RESULTS_PATH = "/home/mila/t/thomas.jiralerspong/llm_compositionality/results"

# Get intrinsic dimension metrics for each model size, for each checkpoint,
# for each level of compositionality of sentences, for each layer


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Create a base directory for all results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_dir = f"{RESULTS_PATH}/pipeline_results_{timestamp}"
ensure_dir(base_dir)

for model_name in MODEL_NAMES:
    df = pd.DataFrame(columns=["index", "model_name", "checkpoint_step", "n_words_correlated", "method", "layer_num", "id"])

    # model_name = model_name.replace("/", "_")   # Replace / with _ to avoid issues with file paths
    # model_dir = os.path.join(base_dir, model_name.replace("/", "_"))
    # ensure_dir(model_dir)

    for checkpoint_step in CHECKPOINT_STEPS:
        # checkpoint_dir = os.path.join(model_dir, f"checkpoint_{checkpoint_step}")
        # ensure_dir(checkpoint_dir)
        model, tokenizer = get_model_and_tokenizer(model_name, checkpoint_step)

        for n_words_correlated in N_WORDS_CORRELATED_LIST:
            data = generate_prompts(N_REPS, n_words_correlated)
            representations = get_reps_from_llm(model, tokenizer, data, DEVICE, BATCH_SIZE)
            ids = calculate_ids(representations, METHODS)

            for key in ids:
                for layer_num, id in enumerate(ids[key]):
                    df.loc[len(df)] = [len(df), model_name, checkpoint_step, n_words_correlated, key, layer_num, id]

    df.to_csv(f"{base_dir}/{model_name.replace('/', '_')}results.csv")

print(f"Results saved in directory: {base_dir}")
