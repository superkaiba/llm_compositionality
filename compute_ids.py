import os
import torch
import pandas as pd
import skdim
from tqdm import tqdm
import time
import argparse

def load_representation(base_path, model_size, checkpoint_step, n_words_correlated):
    # Construct the path to the representation file
    model_name = f"EleutherAI_pythia-{model_size}-deduped"
    file_path = os.path.join(
        base_path,
        model_name,
        f"checkpoint_{checkpoint_step}",
        "representations",
        f"representations_{n_words_correlated}_words_correlated.pt"
    )
    
    # Load the representation
    representation = torch.load(file_path)
    
    return representation

def generate_and_save_id_data(base_path, methods, model_size):
    output_file = os.path.join(base_path, f"results_{model_size}.csv")
    # Check if the output file already exists and load it if it does
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
    else:
        df = pd.DataFrame(columns=['model_size', 'checkpoint_step', 'n_words_correlated', 'layer_num', 'method', 'id'])

    # Get the specific model directory
    model_dir = f"EleutherAI_pythia-{model_size}-deduped"
    model_path = os.path.join(base_path, model_dir)

    if not os.path.exists(model_path):
        print(f"Error: Model directory for size {model_size}b not found.")
        return

    checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith("checkpoint_")]

    for checkpoint_dir in checkpoint_dirs:
        checkpoint_step = int(checkpoint_dir.split('_')[1])
        rep_dir = os.path.join(model_path, checkpoint_dir, "representations")
        rep_files = [f for f in os.listdir(rep_dir) if f.startswith("representations_") and f.endswith("_words_correlated.pt")]

        for rep_file in rep_files:
            n_words_correlated = int(rep_file.split('_')[1])

            representations = torch.load(os.path.join(rep_dir, rep_file))
            representations = [rep.to(torch.float16) for rep in representations]

            for layer_num, layer_rep in enumerate(representations[1:]): # skip the positional embedding layer
                for method_name, method in methods.items():
                    # Check if this combination already exists in the dataframe
                    existing = df[(df['model_size'] == model_size) & 
                                  (df['checkpoint_step'] == checkpoint_step) & 
                                  (df['n_words_correlated'] == n_words_correlated) & 
                                  (df['layer_num'] == layer_num) & 
                                  (df['method'] == method_name)]
                    
                    if existing.empty:
                        start_time = time.time()
                        id_value = method.fit_transform(layer_rep)
                        computation_time = time.time() - start_time
                        print(f"Computed ID: model_size={model_size}, checkpoint_step={checkpoint_step}, n_words_correlated={n_words_correlated}, layer_num={layer_num}, method={method_name}, id={id_value}, time={computation_time:.2f} seconds")
                        new_row = pd.DataFrame({
                            'model_size': [model_size],
                            'checkpoint_step': [checkpoint_step],
                            'n_words_correlated': [n_words_correlated],
                            'layer_num': [layer_num],
                            'method': [method_name],
                            'id': [id_value]
                        })
                        df = pd.concat([df, new_row], ignore_index=True)
                        
                        # Save the dataframe after each new computation
                        df.to_csv(output_file, index=False)
                    else:
                        print(f"Skipping existing ID computation: model_size={model_size}, checkpoint_step={checkpoint_step}, n_words_correlated={n_words_correlated}, layer_num={layer_num}, method={method_name}")
    print(f"All computations complete for model size {model_size}. Final data saved to {output_file}")

METHODS = {
    "mle": skdim.id.MLE(),
    "pca": skdim.id.lPCA(),
    "pr": skdim.id.lPCA(ver="participation_ratio"),
    "2nn": skdim.id.TwoNN(),
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Compute and save ID data for a specific model size.")
    parser.add_argument("--base_path", type=str, required=True, help="Base path for the results directory")
    parser.add_argument("--model_size", type=str, required=True, help="Size of the model to compute (e.g., '6.9')")

    return parser.parse_args()

args = parse_arguments()

generate_and_save_id_data(args.base_path, METHODS, args.model_size)