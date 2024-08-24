
from utils import load_results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.colors as mcolors
from matplotlib import ticker

def load_and_merge_results(ordered_results_dir, shuffled_results_dir):
    import os
    import pandas as pd
    
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
    
    # Save combined results
    output_path = os.path.join(os.path.dirname(ordered_results_dir), 'combined_results.csv')
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
            ax.legend(title='N Words Coupled', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)

        # Remove any unused subplots
        for i in range(max_layer + 1, n_rows * n_cols):
            fig.delaxes(axes[i // 3, i % 3])

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'id_over_checkpoint_{model_size}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    print(f"Plots saved in {save_dir}")


def plot_id_over_layers(combined_results, save_dir):
    print("Plotting ID over layers...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_steps = sorted(combined_results['checkpoint_step'].unique())
    methods = ['pca_ratio_099', '2nn']
    n_words_correlated_list = sorted(combined_results['n_words_correlated'].unique())
    model_sizes = ['410m', '1b', '6.9b']
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_words_correlated_list)))
    
    for checkpoint_step in checkpoint_steps:
        fig, axes = plt.subplots(nrows=len(methods), ncols=3, figsize=(18, 5*len(methods)), squeeze=False)
        fig.suptitle(f'ID over Layers - Checkpoint: {checkpoint_step}', fontsize=20)
        
        for row, method in enumerate(methods):
            for col, model_size in enumerate(model_sizes):
                ax = axes[row, col]
                
                data = combined_results[(combined_results['checkpoint_step'] == checkpoint_step) & 
                                        (combined_results['method'] == method) &
                                        (combined_results['model_size'] == model_size)]
                
                if data.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    continue
                
                for i, n_words in enumerate(n_words_correlated_list):
                    n_words_data = data[data['n_words_correlated'] == n_words]
                    
                    if n_words_data.empty:
                        continue
                    
                    ordered_data = n_words_data[~n_words_data['is_shuffled']]
                    ordered_color = colors[i]
                    ax.plot(ordered_data['layer_num'], ordered_data['id'], 
                            label=f'{n_words} words (ordered)', 
                            color=ordered_color, linestyle='-', marker='o', markersize=4)
                    
                    shuffled_data = n_words_data[n_words_data['is_shuffled']]
                    shuffled_color = tuple(list(ordered_color[:3]) + [0.5])  # Lighter version with 50% opacity
                    ax.plot(shuffled_data['layer_num'], shuffled_data['id'], 
                            label=f'{n_words} words (shuffled)', 
                            color=shuffled_color, linestyle='--', marker='s', markersize=4)
            
                ax.set_title(f'Model: {model_size}, Method: {method}', fontsize=12)
                ax.set_xlabel('Layer Number', fontsize=10)
                ax.set_ylabel('Intrinsic Dimension', fontsize=10)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                if col == 2:
                    ax.legend(title='N Words Correlated', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        save_path_png = os.path.join(save_dir, f'id_over_layers_checkpoint_{checkpoint_step}.png')
        save_path_pdf = os.path.join(save_dir, f'id_over_layers_checkpoint_{checkpoint_step}.pdf')
        plt.savefig(save_path_png, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ID over layers plot for checkpoint {checkpoint_step}:")
        print(f"  - PNG: {save_path_png}")
        print(f"  - PDF: {save_path_pdf}")
    
    print(f"All ID over layers plots saved in {save_dir}")

def plot_id_over_time_per_layer(combined_results, save_dir):
    print("Plotting ID over time per layer...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_sizes = ['410m', '1b', '6.9b']
    methods = ['pca_ratio_099', '2nn']
    n_words_correlated_list = sorted(combined_results['n_words_correlated'].unique())
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_words_correlated_list)))
    
    for model_size in model_sizes:
        num_layers = combined_results[combined_results['model_size'] == model_size]['layer_num'].max() + 1
        
        for layer in range(num_layers):
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), squeeze=False)
            fig.suptitle(f'ID over Time - Layer {layer}', fontsize=16)
            
            for col, model_size in enumerate(model_sizes):
                ax = axes[0, col]
                
                for method in methods:
                    data = combined_results[(combined_results['model_size'] == model_size) & 
                                            (combined_results['method'] == method) &
                                            (combined_results['layer_num'] == layer)]
                    
                    if data.empty:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                        continue
                    
                    for i, n_words in enumerate(n_words_correlated_list):
                        n_words_data = data[data['n_words_correlated'] == n_words]
                        
                        if n_words_data.empty:
                            continue
                        
                        ordered_data = n_words_data[~n_words_data['is_shuffled']]
                        ax.plot(ordered_data['checkpoint_step'], ordered_data['id'], 
                                label=f'{method} - {n_words} words (ordered)', 
                                color=colors[i], linestyle='-', marker='o', markersize=4)
                        
                        shuffled_data = n_words_data[n_words_data['is_shuffled']]
                        ax.plot(shuffled_data['checkpoint_step'], shuffled_data['id'], 
                                label=f'{method} - {n_words} words (shuffled)', 
                                color=colors[i], linestyle='--', marker='s', markersize=4)
                
                ax.set_title(f'Model: {model_size}', fontsize=12)
                ax.set_xlabel('Checkpoint Step', fontsize=10)
                ax.set_ylabel('Intrinsic Dimension', fontsize=10)
                ax.grid(True)
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                if col == 2:
                    ax.legend(title='Method - N Words Correlated', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            
            plt.tight_layout()
            save_path_png = os.path.join(save_dir, f'id_over_time_layer_{layer}.png')
            save_path_pdf = os.path.join(save_dir, f'id_over_time_layer_{layer}.pdf')
            plt.savefig(save_path_png, bbox_inches='tight')
            plt.savefig(save_path_pdf, bbox_inches='tight')
            plt.close()
            
            print(f"Saved ID over time plot for layer {layer}:")
            print(f"  - PNG: {save_path_png}")
            print(f"  - PDF: {save_path_pdf}")
    
    print(f"All ID over time per layer plots saved in {save_dir}")

def plot_id_over_layers_1word(combined_results, save_dir):
    print("Plotting ID over layers for 1 word correlated...")
    
    # Filter data for 1 word correlated
    data = combined_results[combined_results['n_words_correlated'] == 1]
    
    # Get unique model sizes and checkpoints
    model_sizes = sorted(data['model_size'].unique())
    checkpoints = sorted(data['checkpoint_step'].unique())
    
    # Set up colors for different model sizes
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_sizes)))
    
    for checkpoint in checkpoints:
        plt.figure(figsize=(12, 8))
        
        for i, model_size in enumerate(model_sizes):
            model_data = data[(data['model_size'] == model_size) & 
                              (data['checkpoint_step'] == checkpoint)]
            
            if model_data.empty:
                continue
            
            # Plot ordered data
            ordered_data = model_data[~model_data['is_shuffled']]
            plt.plot(ordered_data['layer_num'], ordered_data['id'], 
                     label=f'{model_size} (ordered)', 
                     color=colors[i], linestyle='-', marker='o', markersize=4)
            
            # Plot shuffled data
            shuffled_data = model_data[model_data['is_shuffled']]
            plt.plot(shuffled_data['layer_num'], shuffled_data['id'], 
                     label=f'{model_size} (shuffled)', 
                     color=colors[i], linestyle='--', marker='s', markersize=4)
        
        plt.title(f'ID over Layers - 1 Word Correlated (Checkpoint: {checkpoint})', fontsize=14)
        plt.xlabel('Layer Number', fontsize=12)
        plt.ylabel('Intrinsic Dimension', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Model Size', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        
        save_path_png = os.path.join(save_dir, f'id_over_layers_1word_checkpoint_{checkpoint}.png')
        save_path_pdf = os.path.join(save_dir, f'id_over_layers_1word_checkpoint_{checkpoint}.pdf')
        plt.savefig(save_path_png, bbox_inches='tight')
        plt.savefig(save_path_pdf, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ID over layers plot for checkpoint {checkpoint}:")
        print(f"  - PNG: {save_path_png}")
        print(f"  - PDF: {save_path_pdf}")
    
    print(f"All ID over layers plots for 1 word correlated saved in {save_dir}")

# FIGURE 1
def plot_mean_id_over_model_sizes(combined_results, save_dir):
    print("Plotting mean ID over model sizes...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Define model sizes and their corresponding hidden dimensions
    model_sizes = ['70m', '160m', '410m', '1.4b', '2.8b', '6.9b']
    hidden_dims = {
        '70m': 512, '160m': 768, '410m': 1024, '1.4b': 2048,
        '2.8b': 2560, '6.9b': 4096
    }
    
    methods = ['2nn', 'pca_ratio_099']
    y_labels = {
        '2nn': r"twonn $I_d$",
        'pca_ratio_099': r"pca $d$"
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(combined_results['n_words_correlated']))))
    
    for method, ax in zip(methods, [ax1, ax2]):
        for i, n_words in enumerate(sorted(combined_results['n_words_correlated'].unique())):
            for is_shuffled in [False, True]:
                mean_ids = []
                x_values = []
                for model_size in model_sizes:
                    data = combined_results[
                        (combined_results['model_size'] == model_size) & 
                        (combined_results['method'] == method) &
                        (combined_results['is_shuffled'] == is_shuffled) &
                        (combined_results['n_words_correlated'] == n_words)
                    ]
                    if not data.empty:
                        mean_ids.append(data['id'].mean())
                        x_values.append(hidden_dims[model_size])
                
                label = f'{n_words} words ({"shuffled" if is_shuffled else "sane"})'
                linestyle = ':' if is_shuffled else '-'
                marker = 's' if is_shuffled else 'o'
                
                ax.plot(x_values, mean_ids, label=label, 
                        linestyle=linestyle, color=colors[i], marker=marker, markersize=6)
        
        ax.set_xlabel('hidden dimension')
        ax.set_ylabel(y_labels[method])
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.legend(title='Words coupled', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path_png = os.path.join(save_dir, 'mean_id_over_model_sizes.png')
    save_path_pdf = os.path.join(save_dir, 'mean_id_over_model_sizes.pdf')
    plt.savefig(save_path_png, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Saved mean ID over model sizes plot:")
    print(f"  - PNG: {save_path_png}")
    print(f"  - PDF: {save_path_pdf}")

def plot_id_over_layers(combined_results, save_dir):
    print("Plotting ID over layers...")
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    model_sizes = ['410m', '1.4b', '6.9b']
    methods = ['2nn', 'pca_ratio_099']
    y_labels = {
        '2nn': r"twonn $I_d$",
        'pca_ratio_099': r"pca $d$"
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharex=False)
    
    # Filter for the final checkpoint step
    final_checkpoint = 143000
    final_data = combined_results[combined_results['checkpoint_step'] == final_checkpoint]
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 4))  # 4 colors for 1-4 words
    
    for col, model_size in enumerate(model_sizes):
        for row, method in enumerate(methods):
            ax = axes[row, col]
            
            max_layer = 0
            for i, n_words in enumerate(range(1, 5)):
                for is_shuffled in [False, True]:
                    data = final_data[
                        (final_data['model_size'] == model_size) & 
                        (final_data['method'] == method) &
                        (final_data['is_shuffled'] == is_shuffled) &
                        (final_data['n_words_correlated'] == n_words)
                    ]
                    
                    if not data.empty:
                        label = f'{n_words} words'
                        linestyle = ':' if is_shuffled else '-'
                        ax.plot(data['layer_num'], data['id'], label=label, linestyle=linestyle, 
                                color=colors[i])
                        max_layer = max(max_layer, data['layer_num'].max())
            
            ax.set_xlabel('layer')
            ax.set_ylabel(y_labels[method])
            ax.set_title(f"{model_size}")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, max_layer)
            
            if row == 0 and col == 2:
                ax.legend(title='Words coupled', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_path_png = os.path.join(save_dir, f'id_over_layers_checkpoint_{final_checkpoint}.png')
    save_path_pdf = os.path.join(save_dir, f'id_over_layers_checkpoint_{final_checkpoint}.pdf')
    plt.savefig(save_path_png, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    plt.close()
    
    print(f"Saved ID over layers plot for checkpoint {final_checkpoint}:")
    print(f"  - PNG: {save_path_png}")
    print(f"  - PDF: {save_path_pdf}")

# We want to plot the id over time for each model size, n_words_correlated, and method
if __name__ == "__main__":
    combined_results = load_and_merge_results(
        "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/final_ordered_20240815_013151", 
        "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/final_shuffled_20240815_013501"
        )
    plot_mean_id_over_model_sizes(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots/final_plots")
    # plot_id_over_layers(combined_results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/plots/final_plots")