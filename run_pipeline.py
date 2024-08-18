import json
import os
from datetime import datetime
import skdim
import sklearn
from dadapy import Data
import argparse
import numpy as np
from utils import *
from id_measures import *
import pandas as pd
import os
import torch


os.environ['HF_HOME'] = "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
os.environ['TRANSFORMERS_CACHE'] = "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/hugging_face_cache"
# MODEL_SIZES = ["70m", "160m", "410m" , "1b", "1.4b", "2.8b", "6.9b", "12b"]
# MODEL_SIZES = ["2.8b"]# , "6.9b", "12b"]
# MODEL_SIZES = [ "6.9b", "1.4b", "2.8b", "12b"]
# MODEL_SIZES = ["6.9b"]
# MODEL_SIZES = ["6.9b"]
# MODEL_SIZES = ["12b"]
# CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# CHECKPOINT_STEPS += [1000 * i for i in range(1, 144)]
DATA_DIR = "/home/mila/t/thomas.jiralerspong/llm_compositionality/data"

CHECKPOINT_STEPS = [0, 2, 64, 512, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
# CHECKPOINT_STEPS = [1, 4, 8,16, 32, 128, 256,1000,3000,13000,53000, 103000,133000]
N_WORDS_CORRELATED_LIST = [i for i in range(1, 5)]
DEVICE = 'cuda'
BATCH_SIZE = 64
METHODS = {
  "mle": skdim.id.MLE(),
  "pca": PCA(),
  "pr": ParticipationRatio(),
  "2nn": skdim.id.TwoNN(),
} 

RESULTS_PATH = "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results"

# Get intrinsic dimension metrics for each model size, for each checkpoint,
# for each level of compositionality of sentences, for each layer

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline with optional prompt shuffling.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the prompts")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--model_size", type=str, help="Model size")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    return parser.parse_args()

args = parse_args()
model_name = f"EleutherAI/pythia-{args.model_size}-deduped" 

# Create a base directory for all results
# shuffle_str = "shuffled" if args.shuffle else "ordered"
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

base_dir = f"{RESULTS_PATH}/{args.exp_name}"
ensure_dir(base_dir)


df = pd.DataFrame(columns=["index", "model_name", "checkpoint_step", "n_words_correlated", "method", "layer_num", "id"])

    # model_name = model_name.replace("/", "_")   # Replace / with _ to avoid issues with file paths
    # model_dir = os.path.join(base_dir, model_name.replace("/", "_"))
    # ensure_dir(model_dir)

for checkpoint_step in CHECKPOINT_STEPS:
    # checkpoint_dir = os.path.join(model_dir, f"checkpoint_{checkpoint_step}")
    # ensure_dir(checkpoint_dir)
    model, tokenizer = get_model_and_tokenizer(model_name, checkpoint_step)

    for n_words_correlated in N_WORDS_CORRELATED_LIST:
        train_data, test_data = load_prompts(n_words_correlated, DATA_DIR, shuffle=args.shuffle)
        data = train_data + test_data
        if args.debug:
            data = data[:1000]
        with torch.no_grad():
            representations = get_reps_from_llm(model, tokenizer, data, DEVICE, BATCH_SIZE)
            # ids = calculate_ids(representations, METHODS)

        # Save the representations
        rep_dir = os.path.join(base_dir, model_name.replace("/", "_"), f"checkpoint_{checkpoint_step}", "representations")
        ensure_dir(rep_dir)
        rep_filename = f"representations_{n_words_correlated}_words_correlated.pt"
        rep_path = os.path.join(rep_dir, rep_filename)
        torch.save(representations, rep_path)
        # for key in ids:
        #     for layer_num, id in enumerate(ids[key]):
        #         df.loc[len(df)] = [len(df), model_name, checkpoint_step, n_words_correlated, key, layer_num, id]

# results_dir = os.path.join(base_dir, model_name.replace("/", "_"), "results.csv")
# df.to_csv(results_dir)

print(f"Results saved in directory: {base_dir}")
