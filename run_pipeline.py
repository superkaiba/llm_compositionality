import json
import os
from datetime import datetime
import sklearn
import argparse
import numpy as np
from utils import *
from id_measures import *
import pandas as pd
import os
import torch


# os.environ['HF_HOME'] = "/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS
# os.environ['TRANSFORMERS_CACHE'] = "/home/echeng/scratch/hugging_face_cache" # TODO: EMILY CHANGE THIS

# CHECKPOINT_STEPS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
# CHECKPOINT_STEPS += [1000 * i for i in range(1, 144)]

# CHECKPOINT_STEPS = [0, 2, 64, 512, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
# CHECKPOINT_STEPS = [1, 4, 8,16, 32, 128, 256,1000,3000,13000,53000, 103000,133000]

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the pipeline with optional prompt shuffling.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the prompts")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--model_size", type=str, help="Model size")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--checkpoint_steps", type=int, nargs="+", help="Checkpoint steps")
    parser.add_argument("--n_words_correlated", type=int, nargs="+", help="Number of words correlated")
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--device", type=str, help="Device", default="cuda")
    parser.add_argument("--data_dir", type=str, help="Data directory", default="/home/echeng/llm_compositionality/data")
    parser.add_argument("--results_path", type=str, help="Results path", default="/home/echeng/llm_compositionality/results_new")

    return parser.parse_args()

args = parse_args()
print(args)

model_name = f"EleutherAI/pythia-{args.model_size}-deduped" if args.model_size not in ('14m',) else f"EleutherAI/pythia-{args.model_size}"

base_dir = f"{args.results_path}"
ensure_dir(base_dir)


df = pd.DataFrame(columns=["index", "model_name", "checkpoint_step", "n_words_correlated", "method", "layer_num", "id"])

for checkpoint_step in args.checkpoint_steps:
    model, tokenizer = get_model_and_tokenizer(model_name, checkpoint_step)

    for n_words_correlated in args.n_words_correlated:
        for rs in range(5):
            rep_dir = os.path.join(base_dir, model_name.replace("/", "_"), f"checkpoint_{checkpoint_step}", "representations")
            ensure_dir(rep_dir)
            rep_filename = f'{"shuffled" if args.shuffle else "sane"}_representations_{n_words_correlated}_words_correlated_rs{rs}.pt'
            rep_path = os.path.join(rep_dir, rep_filename)

            # we already did it.
            if os.path.exists(rep_path): 
                print('already done')
                continue

            train_data, test_data = load_prompts(n_words_correlated, args.data_dir, shuffle=args.shuffle, rs=rs)
            data = train_data + test_data

            if args.debug:
                data = data[:1000]
                
            with torch.no_grad():
                representations = get_reps_from_llm(model, tokenizer, data, args.device, args.batch_size)

            # Save the representations
            torch.save(representations, rep_path)

            del representations 
    
    del model
    del tokenizer

print(f"Results saved in directory: {base_dir}")
