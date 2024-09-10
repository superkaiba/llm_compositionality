from utils import load_results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.colors as mcolors
from matplotlib import ticker

def load_and_merge_results(ordered_results_dir, shuffled_results_dir, recompute=False):
    import os
    import pandas as pd
    # Save combined results
    output_path = os.path.join(os.path.dirname(ordered_results_dir), 'combined_results.csv')
    if os.path.exists(output_path):
        print(f"Combined results already exist at {output_path}. Skipping loading and merging.")
        return pd.read_csv(output_path)

    all_results = []
    
    # Process ordered results
    for filename in os.listdir(ordered_results_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(ordered_results_dir, filename)
            df = pd.read_csv(file_path)
            df['is_shuffled'] = False
            all_results.append(df)
    
    # Process shuffled results
    for filename in os.listdir(shuffled_results_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(shuffled_results_dir, filename)
            df = pd.read_csv(file_path)
            df['is_shuffled'] = True
            all_results.append(df)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Sort the combined results
    combined_results = combined_results.sort_values(
        by=['model_size', 'checkpoint_step', 'n_words_correlated', 'layer_num']
    )
    
    combined_results.to_csv(output_path, index=False)
    
    return combined_results
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_id_over_checkpoint(combined_results, save_dir):

    # Get unique values for grouping
    model_sizes = combined_results['model_size'].unique()
    methods = combined_results['method'].unique()
    n_words_correlated = combined_results['n_words_correlated'].unique()

    # Set up color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_words_correlated)))

    for model_size in model_sizes:
        model_data = combined_results[combined_results['model_size'] == model_size]
        
        # Get the maximum layer for this specific model size
        max_layer = model_data['layer_num'].max()
        
        # Calculate the number of rows and columns for subplots
        n_rows = (max_layer + 1 + 2) // 3  # +2 to round up
        n_cols = 3

        # Create a subplot for each layer
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5*n_rows), squeeze=False)
        fig.suptitle(f'ID over Checkpoint Step for Model Size: {model_size}', fontsize=16)
        print("Model size: ", model_size)
        print("Max layer: ", max_layer)
        for layer in range(max_layer + 1):
            ax = axes[layer // 3, layer % 3]
            layer_data = model_data[model_data['layer_num'] == layer]

            for method in methods:
                method_data = layer_data[layer_data['method'] == method]

                for i, n_words in enumerate(n_words_correlated):
                    data = method_data[method_data['n_words_correlated'] == n_words]
                    
                    # Plot ordered data
                    ordered_data = data[~data['is_shuffled']]
                    ax.plot(ordered_data['checkpoint_step'], ordered_data['id'], 
                            label=f'{n_words} words (ordered)', color=colors[i], linestyle='-')
                    
                    # Plot shuffled data
                    shuffled_data = data[data['is_shuffled']]
                    ax.plot(shuffled_data['checkpoint_step'], shuffled_data['id'], 
                            label=f'{n_words} words (shuffled)', color=colors[i], linestyle='--')

            ax.set_title(f'Layer {layer}')
            ax.set_xlabel('Checkpoint Step')
            ax.set_ylabel('Intrinsic Dimension')
            ax.legend(title='N Words Correlated', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

        # Remove any unused subplots
        for i in range(max_layer + 1, n_rows * n_cols):
            fig.delaxes(axes[i // 3, i % 3])

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'id_over_checkpoint_{model_size}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Plots saved in {save_dir}")

def plot_final_layer_id_over_time(combined_results, save_dir):
    # Filter for the final layer of each model size
    max_layers = combined_results.groupby('model_size')['layer_num'].max()
    final_layer_data = combined_results[combined_results.apply(lambda row: row['layer_num'] == max_layers[row['model_size']], axis=1)]

    # Get unique values for grouping
    model_sizes = final_layer_data['model_size'].unique()
    methods = final_layer_data['method'].unique()
    n_words_correlated = final_layer_data['n_words_correlated'].unique()

    # Set up color palette for model sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_sizes)))

    # Create a single figure with subplots for each method
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    fig.suptitle('ID in Final Layer Over Time for All Model Sizes', fontsize=20)

    for idx, method in enumerate(methods):
        ax = axes[idx // 2, idx % 2]
        method_data = final_layer_data[final_layer_data['method'] == method]
        
        for i, model_size in enumerate(model_sizes):
            model_method_data = method_data[method_data['model_size'] == model_size]
            
            for n_words in n_words_correlated:
                data = model_method_data[model_method_data['n_words_correlated'] == n_words]
                
                # Plot ordered data
                ordered_data = data[~data['is_shuffled']]
                ax.plot(ordered_data['checkpoint_step'], ordered_data['id'], 
                        label=f'{model_size} - {n_words} words (ordered)', 
                        color=colors[i], linestyle='-')
                
                # Plot shuffled data
                shuffled_data = data[data['is_shuffled']]
                ax.plot(shuffled_data['checkpoint_step'], shuffled_data['id'], 
                        label=f'{model_size} - {n_words} words (shuffled)', 
                        color=colors[i], linestyle='--')

        ax.set_title(f'{method.upper()} method', fontsize=16)
        ax.set_xlabel('Checkpoint Step', fontsize=12)
        ax.set_ylabel('Intrinsic Dimension', fontsize=12)
        ax.grid(True)

    # Create a single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, title='Model Size - N Words Correlated', 
               bbox_to_anchor=(1.05, 0.5), loc='center left')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'final_layer_id_over_time_all_methods.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Final layer plot saved in {save_dir}")

def plot_id_over_layers(combined_results, save_dir):
    print("Plotting ID over layers...")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique values for each parameter
    checkpoint_steps = sorted(combined_results['checkpoint_step'].unique())
    methods = combined_results['method'].unique()
    n_words_correlated_list = sorted(combined_results['n_words_correlated'].unique())
    model_sizes = combined_results['model_size'].unique()# Sort by numeric value
    
    # Set up color palette for n_words_correlated
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_words_correlated_list)))
    
    for model_size in model_sizes:
        # Create a single figure with subplots for each method and checkpoint
        n_rows = len(methods)
        n_cols = len(checkpoint_steps)
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
        fig.suptitle(f'ID over Layers - Model: {model_size}', fontsize=20)
        
        for row, method in enumerate(methods):
            for col, checkpoint_step in enumerate(checkpoint_steps):
                ax = axes[row, col]
                
                # Filter data for current checkpoint, method, and model size
                data = combined_results[(combined_results['checkpoint_step'] == checkpoint_step) & 
                                        (combined_results['method'] == method) &
                                        (combined_results['model_size'] == model_size)]
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    continue
                
                for i, n_words in enumerate(n_words_correlated_list):
                    # Filter data for current n_words_correlated
                    n_words_data = data[data['n_words_correlated'] == n_words]
                    
                    if n_words_data.empty:
                        continue
                    
                    # Plot ordered data
                    ordered_data = n_words_data[~n_words_data['is_shuffled']]
                    ax.plot(ordered_data['layer_num'], ordered_data['id'], 
                            label=f'{n_words} words (ordered)', 
                            color=colors[i], linestyle='-', marker='o', markersize=4)
                    
                    # Plot shuffled data
                    shuffled_data = n_words_data[n_words_data['is_shuffled']]
                    ax.plot(shuffled_data['layer_num'], shuffled_data['id'], 
                            label=f'{n_words} words (shuffled)', 
                            color=colors[i], linestyle='--', marker='s', markersize=4)
                
                ax.set_title(f'{method.upper()} - Checkpoint: {checkpoint_step}', fontsize=12)
                ax.set_xlabel('Layer Number', fontsize=10)
                ax.set_ylabel('Intrinsic Dimension', fontsize=10)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Only add legend to the first subplot
                if row == 0 and col == 0:
                    ax.legend(title='N Words Correlated', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        save_path_png = os.path.join(save_dir, f'id_over_layers_{model_size}.png')
        save_path_pdf = os.path.join(save_dir, f'id_over_layers_{model_size}.pdf')
        plt.savefig(save_path_png, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ID over layers plot for {model_size} model as PNG: {save_path_png}")
        print(f"Saved ID over layers plot for {model_size} model as PDF: {save_path_pdf}")
    
    print(f"All ID over layers plots saved in {save_dir}")

def plot_id_over_time_per_layer(combined_results, save_dir):
    print("Plotting ID over time per layer...")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique values for each parameter
    model_sizes = combined_results['model_size'].unique()
    methods = combined_results['method'].unique()
    n_words_correlated_list = sorted(combined_results['n_words_correlated'].unique())
    
    # Color palette for n_words_correlated
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_words_correlated_list)))
    
    for model_size in model_sizes:
        # Get number of layers
        num_layers = combined_results[combined_results['model_size'] == model_size]['layer_num'].max() + 1
        
        # Create subplots
        fig, axes = plt.subplots(nrows=num_layers, ncols=len(methods), figsize=(6*len(methods), 4*num_layers), squeeze=False)
        fig.suptitle(f'ID over Time - {model_size}', fontsize=16)
        
        for layer in range(num_layers):
            for col, method in enumerate(methods):
                ax = axes[layer, col]
                
                # Filter data for current model size, method, and layer
                data = combined_results[(combined_results['model_size'] == model_size) & 
                                        (combined_results['method'] == method) &
                                        (combined_results['layer_num'] == layer)]
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    continue
                
                for i, n_words in enumerate(n_words_correlated_list):
                    # Filter data for current n_words_correlated
                    n_words_data = data[data['n_words_correlated'] == n_words]
                    
                    if n_words_data.empty:
                        continue
                    
                    # Plot ordered data
                    ordered_data = n_words_data[~n_words_data['is_shuffled']]
                    ax.plot(ordered_data['checkpoint_step'], ordered_data['id'], 
                            label=f'{n_words} words (ordered)', 
                            color=colors[i], linestyle='-', marker='o', markersize=4)
                    
                    # Plot shuffled data
                    shuffled_data = n_words_data[n_words_data['is_shuffled']]
                    ax.plot(shuffled_data['checkpoint_step'], shuffled_data['id'], 
                            label=f'{n_words} words (shuffled)', 
                            color=colors[i], linestyle='--', marker='s', markersize=4)
                
                ax.set_title(f'{method.upper()} - Layer {layer}', fontsize=12)
                ax.set_xlabel('Checkpoint Step', fontsize=10)
                ax.set_ylabel('Intrinsic Dimension', fontsize=10)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                # Only add legend to the first subplot
                if layer == 0 and col == 0:
                    ax.legend(title='N Words Correlated', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        save_path_png = os.path.join(save_dir, f'id_over_time_per_layer_{model_size}.png')
        save_path_pdf = os.path.join(save_dir, f'id_over_time_per_layer_{model_size}.pdf')
        plt.savefig(save_path_png, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ID over time per layer plot for {model_size} model:")
        print(f"  - PNG: {save_path_png}")
        print(f"  - PDF: {save_path_pdf}")
    
    print(f"All ID over time per layer plots saved in {save_dir}")

def plot_min_max_id_over_model_sizes(combined_results, save_dir):
    # Filter for the final checkpoint step
    final_checkpoint = combined_results['checkpoint_step'].max()
    final_data = combined_results[combined_results['checkpoint_step'] == final_checkpoint]

    # Define model sizes and their numeric values
    model_sizes = ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']
    model_sizes_numeric = [70_000_000, 160_000_000, 410_000_000, 1_000_000_000, 1_400_000_000, 2_800_000_000, 6_900_000_000]
    
    # Filter data for only the model sizes we're interested in
    final_data = final_data[final_data['model_size'].isin(model_sizes)]

    methods = final_data['method'].unique()
    n_words_correlated_list = sorted(final_data['n_words_correlated'].unique())

    # Set up the plots
    fig, axes = plt.subplots(len(methods), 3, figsize=(24, 8*len(methods)), sharex='col')
    if len(methods) == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.rainbow(np.linspace(0, 1, len(n_words_correlated_list)))

    for method_idx, method in enumerate(methods):
        method_data = final_data[final_data['method'] == method]

        for i, n_words in enumerate(n_words_correlated_list):
            n_words_data = method_data[method_data['n_words_correlated'] == n_words]

            for shuffle_status in [False, True]:
                data = n_words_data[n_words_data['is_shuffled'] == shuffle_status]
                
                min_ids = []
                max_ids = []
                mean_ids = []
                for size in model_sizes:
                    size_data = data[data['model_size'] == size]
                    if not size_data.empty:
                        min_ids.append(size_data['id'].min())
                        max_ids.append(size_data['id'].max())
                        mean_ids.append(size_data['id'].mean())
                    else:
                        min_ids.append(np.nan)
                        max_ids.append(np.nan)
                        mean_ids.append(np.nan)

                label = f'{n_words} words ({"Shuffled" if shuffle_status else "Ordered"})'
                linestyle = '--' if shuffle_status else '-'
                
                # Plot min ID for model sizes
                axes[method_idx, 0].plot(model_sizes_numeric, min_ids, marker='o', linestyle=linestyle, color=colors[i], 
                                         label=label, alpha=0.7)
                
                # Plot max ID for model sizes
                axes[method_idx, 1].plot(model_sizes_numeric, max_ids, marker='s', linestyle=linestyle, color=colors[i], 
                                         label=label, alpha=0.7)
                
                # Plot mean ID for model sizes
                axes[method_idx, 2].plot(model_sizes_numeric, mean_ids, marker='^', linestyle=linestyle, color=colors[i], 
                                         label=label, alpha=0.7)

        for col in range(3):
            ax = axes[method_idx, col]
            ax.set_title(f'{method.upper()} - {"Min" if col == 0 else "Max" if col == 1 else "Mean"} ID at Final Checkpoint', fontsize=14)
            ax.set_ylabel('Intrinsic Dimension', fontsize=12)
            ax.set_yscale('linear')
            ax.set_xscale('linear')  # Changed to linear scale
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlabel('Model Size (number of parameters)', fontsize=12)
            ax.set_xticks(model_sizes_numeric)
            ax.set_xticklabels(model_sizes, rotation=45)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    save_path_png = os.path.join(save_dir, 'min_max_mean_id_over_model_sizes.png')
    save_path_pdf = os.path.join(save_dir, 'min_max_mean_id_over_model_sizes.pdf')
    plt.savefig(save_path_png, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()

    print(f"Saved Min-Max-Mean ID over model sizes plot:")
    print(f"  - PNG: {save_path_png}")
    print(f"  - PDF: {save_path_pdf}")

def plot_min_max_id_over_hidden_dimensions(combined_results, save_dir):
    print("Plotting Min-Max-Mean ID over hidden dimensions...")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Define the mapping of model sizes to hidden dimensions
    model_size_to_dim = {
        '70m': 512,
        '160m': 768,
        '410m': 1024,
        '1.4b': 2048,
        '2.8b': 2560,
        '6.9b': 4096
    }
    
    # Filter the data for the specified model sizes
    filtered_results = combined_results[combined_results['model_size'].isin(model_size_to_dim.keys())]
    
    # Get unique values for grouping
    methods = filtered_results['method'].unique()
    n_words_correlated = filtered_results['n_words_correlated'].unique()
    hidden_dimensions = sorted([model_size_to_dim[size] for size in model_size_to_dim.keys()])
    
    # Set up color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_words_correlated)))
    
    # Create a figure with subplots for each method
    fig, axes = plt.subplots(nrows=len(methods), ncols=3, figsize=(30, 8*len(methods)), squeeze=False)
    fig.suptitle('ID over Hidden Dimensions', fontsize=16)
    
    for method_idx, method in enumerate(methods):
        method_data = filtered_results[filtered_results['method'] == method]
        
        for i, n_words in enumerate(n_words_correlated):
            n_words_data = method_data[method_data['n_words_correlated'] == n_words]
            
            for shuffle_status in [False, True]:
                data = n_words_data[n_words_data['is_shuffled'] == shuffle_status]
                
                min_ids = []
                max_ids = []
                mean_ids = []
                for dim in hidden_dimensions:
                    size = next(size for size, d in model_size_to_dim.items() if d == dim)
                    size_data = data[data['model_size'] == size]
                    if not size_data.empty:
                        min_ids.append(size_data['id'].min())
                        max_ids.append(size_data['id'].max())
                        mean_ids.append(size_data['id'].mean())
                    else:
                        min_ids.append(np.nan)
                        max_ids.append(np.nan)
                        mean_ids.append(np.nan)

                label = f'{n_words} words ({"Shuffled" if shuffle_status else "Ordered"})'
                linestyle = '--' if shuffle_status else '-'
                
                # Plot min ID for hidden dimensions
                axes[method_idx, 0].plot(hidden_dimensions, min_ids, marker='o', linestyle=linestyle, color=colors[i], 
                                         label=label, alpha=0.7)
                
                # Plot max ID for hidden dimensions
                axes[method_idx, 1].plot(hidden_dimensions, max_ids, marker='s', linestyle=linestyle, color=colors[i], 
                                         label=label, alpha=0.7)
                
                # Plot mean ID for hidden dimensions
                axes[method_idx, 2].plot(hidden_dimensions, mean_ids, marker='^', linestyle=linestyle, color=colors[i], 
                                         label=label, alpha=0.7)

        for col in range(3):
            ax = axes[method_idx, col]
            ax.set_title(f'{method.upper()} - {"Min" if col == 0 else "Max" if col == 1 else "Mean"} ID at Final Checkpoint', fontsize=14)
            ax.set_ylabel('Intrinsic Dimension', fontsize=12)
            ax.set_yscale('linear')
            ax.set_xscale('linear')  # Changed to linear scale
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlabel('Model Hidden Dimension', fontsize=12)
            ax.set_xticks(hidden_dimensions)
            ax.set_xticklabels(hidden_dimensions, rotation=45)
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    plt.tight_layout()
    save_path_png = os.path.join(save_dir, 'min_max_mean_id_over_hidden_dimensions.png')
    save_path_pdf = os.path.join(save_dir, 'min_max_mean_id_over_hidden_dimensions.pdf')
    plt.savefig(save_path_png, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()

    print(f"Saved Min-Max-Mean ID over hidden dimensions plot:")
    print(f"  - PNG: {save_path_png}")
    print(f"  - PDF: {save_path_pdf}")


# We want to plot the id over time for each model size, n_words_correlated, and method
if __name__ == "__main__":
    combined_results = load_and_merge_results(
        "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/final_ordered_20240815_013151", 
        "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/final_shuffled_20240815_013501"
        )
    plot_min_max_id_over_model_sizes(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots")
    plot_min_max_id_over_hidden_dimensions(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots")
    # plot_id_over_checkpoint(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots")
    # plot_final_layer_id_over_time(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots")
    # # plot_compositionality_over_time(args.results_path, args.save_dir)
    # plot_id_over_time_per_layer(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots")
    # plot_id_over_layers(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots")

# python plot.py --results_path /home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/pipeline_results_20240811_141329/EleutherAI_pythia-1.4b-dedupedresults.csv --save_dir /home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/pipeline_results_20240811_141329