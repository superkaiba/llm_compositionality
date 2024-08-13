from utils import load_results
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.colors as mcolors
from matplotlib import ticker
def plot_results(results, save_dir):

    fig, ax = plt.subplots(1, 1)
    for model_name in results:
        for checkpoint in results[model_name]:
            
            words_correlated_results = [results[model_name][checkpoint][n_words_correlated]['mle'][0] for n_words_correlated in results[model_name][checkpoint]]
            for n_words_correlated in results[model_name][checkpoint]:
                ax.plot(results[model_name][checkpoint][int(n_words_correlated)]['mle'], label=f"{model_name}_{checkpoint}_{n_words_correlated}")
    
    ax.legend()
    ax.set_xlabel('Layer')
    ax.set_ylabel('ID')
    plt.savefig(f"{save_dir}/results.png")  
    

def plot_compositionality_over_time(csv_file, save_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    layer_numbers = df['layer_num'].unique()
    n_layers = len(layer_numbers)

    # Calculate the number of rows and columns for subplots
    n_cols = min(4, n_layers)  # Limit to 4 columns max
    n_rows = (n_layers + n_cols - 1) // n_cols

    # Set up the plot style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Create subplots with a smaller figure size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 1.7 * n_rows))
    fig.suptitle("Compositionality Over Time", fontsize=16, y=1.02)

    # Flatten the axes array for easier indexing
    axes = axes.flatten() if n_layers > 1 else [axes]

    for idx, layer_number in enumerate(layer_numbers):
        ax = axes[idx]
        layer_df = df[df['layer_num'] == layer_number]
        n_words_correlated = layer_df['n_words_correlated'].unique()

        for n_words in n_words_correlated:
            n_words_correlated_df = layer_df[layer_df['n_words_correlated'] == n_words]
            ax.plot(n_words_correlated_df['checkpoint_step'], n_words_correlated_df['id'], 
                    label=f"{n_words}", linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Training epoch', fontsize=8)
        ax.set_ylabel('ID', fontsize=8)
        ax.set_title(f"Layer {layer_number}", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, linestyle='--', alpha=0.5)
    # Set x-axis to log scale for each subplot
    for ax in axes:
        ax.set_xscale('log')
        
        # Adjust x-axis ticks for better readability in log scale
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        
        # Rotate x-axis labels for better fit
        ax.tick_params(axis='x', rotation=45)
    # Remove any unused subplots
    for idx in range(n_layers, len(axes)):
        fig.delaxes(axes[idx])

    # Add a common legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', title='Number of correlated words', 
               ncol=len(n_words_correlated), bbox_to_anchor=(0.5, 1.05), 
               fontsize=8, title_fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{csv_file.split('/')[-1].replace('/', '_')}_compositionality_over_time.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_compositionality_gap_over_layers(csv_file, save_dir):
    # Load the CSV data
    df = pd.read_csv(csv_file)

    # Group the data by model_name, checkpoint_step, method, and layer_num
    grouped = df.groupby(['model_name', 'checkpoint_step', 'method', 'layer_num'])

    # Calculate the difference in intrinsic dimension between the two n_words_correlated values
    diff_df = grouped.apply(lambda x: x.loc[x['n_words_correlated'].idxmin(), 'id']
                                    - x.loc[x['n_words_correlated'].idxmax(), 'id'])
    diff_df = diff_df.reset_index(name='intrinsic_dimension_gap')

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.set_style("whitegrid")

    # Get the range of checkpoint steps for normalization
    min_step = diff_df['checkpoint_step'].min()
    max_step = diff_df['checkpoint_step'].max()

    # Create a custom colormap from light blue to dark blue
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["lightblue", "darkblue"])
    norm = mcolors.Normalize(vmin=min_step, vmax=max_step)

    # Plot lines with varying color based on checkpoint step
    for (model, method), group in diff_df.groupby(['model_name', 'method']):
        sorted_group = group.sort_values(['checkpoint_step', 'layer_num'])
        for checkpoint, checkpoint_group in sorted_group.groupby('checkpoint_step'):
            color = cmap(norm(checkpoint))
            plt.plot(checkpoint_group['layer_num'], checkpoint_group['intrinsic_dimension_gap'], 
                     color=color, marker='o', markersize=4, linewidth=1.5)

    # Add a single legend entry for each model-method combination
    for (model, method) in diff_df.groupby(['model_name', 'method']).groups.keys():
        plt.plot([], [], color='gray', label=f"{model} - {method}")

    plt.xlabel('Layer Number')
    plt.ylabel('Intrinsic Dimension Gap')
    plt.title('Gap in Intrinsic Dimension vs. Layer Number')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Checkpoint Step', pad=0.1)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{csv_file.split('/')[-1].replace('/', '_')}_compositionality_gap_over_layers.png", 
                bbox_inches='tight', dpi=300)
    plt.close(fig)

def convert_to_number(s):
    """
    Converts a string representing a number with 'm' (millions) or 'b' (billions) 
    into its numerical form with commas as thousand separators.

    Args:
    s (str): The string to convert, e.g., "70m", "1.6b"

    Returns:
    str: The numerical form of the string, e.g., "70,000,000", "1,600,000,000"
    """
    multiplier = 1
    
    # Determine the multiplier based on the suffix
    if 'm' in s.lower():
        multiplier = 1_000_000
        s = s.lower().replace('m', '')
    elif 'b' in s.lower():
        multiplier = 1_000_000_000
        s = s.lower().replace('b', '')
    
    # Convert the remaining string to a float and apply the multiplier
    number = float(s) * multiplier
    
    # Convert to integer and return formatted with commas
    return int(number)

def plot_id_over_sizes(save_dir):
    import glob

    # Function to extract model size from model name
    def extract_model_size(model_name):
        return convert_to_number(model_name.split('-')[1])
        
    # Read all CSV files
    csv_files = glob.glob(f'{save_dir}/*.csv')
    data = []

    for file in csv_files:
        df = pd.read_csv(file)
        
        # Get the model name and size
        model_name = df['model_name'].iloc[0]
        model_size = extract_model_size(model_name)
        
        # Get the ID of the final layer
        final_layer_id = df[df['layer_num'] == df['layer_num'].max()]['id'].iloc[0]
        
        data.append({'model_size': model_size, 'final_layer_id': final_layer_id})

    # Create a DataFrame from the collected data
    plot_df = pd.DataFrame(data)

    # Sort by model size
    plot_df = plot_df.sort_values('model_size')

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df['model_size'], plot_df['final_layer_id'], marker='o')
    plt.xlabel('Model Size')
    plt.ylabel('Final Layer ID')
    plt.title('Final Layer ID vs Model Size')
    plt.xscale('log')  # Use log scale for x-axis if model sizes vary greatly
    plt.grid(True)

    # Add annotations for each point
    for i, row in plot_df.iterrows():
        plt.annotate(f"{row['model_size']}", (row['model_size'], row['final_layer_id']), 
                    textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_layer_id_vs_model_size.png") 


def plot_id_diff_over_sizes(save_dir):
    import glob

    # Function to extract model size from model name
    def extract_model_size(model_name):
        return convert_to_number(model_name.split('-')[1])
        
    # Read all CSV files
    csv_files = glob.glob(f'{save_dir}/*.csv')
    data = []

    for file in csv_files:
        df = pd.read_csv(file)
        
        # Get the model name and size
        model_name = df['model_name'].iloc[0]
        model_size = extract_model_size(model_name)
        
        # Get the ID of the final layer
        final_layer_df = df[df['layer_num'] == df['layer_num'].max()]
        max_id = final_layer_df['id'].max()
        min_id = final_layer_df['id'].min()
        
        final_layer_id_diff = max_id - min_id
        
        data.append({'model_size': model_size, 'final_layer_id_diff': final_layer_id_diff})

    # Create a DataFrame from the collected data
    plot_df = pd.DataFrame(data)
    # Sort by model size
    plot_df = plot_df.sort_values('model_size')

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df['model_size'], plot_df['final_layer_id_diff'], marker='o')
    plt.xlabel('Model Size')
    plt.ylabel('Final Layer ID Diff')
    plt.title('Final Layer ID Diff vs Model Size')
    plt.xscale('log')  # Use log scale for x-axis if model sizes vary greatly
    plt.grid(True)

    # Add annotations for each point
    for i, row in plot_df.iterrows():
        plt.annotate(f"{row['model_size']}", (row['model_size'], row['final_layer_id_diff']), 
                    textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/final_layer_id_diff_vs_model_size.png") 

def plot_intrinsic_dimension_over_layers(csv_file, checkpoint_step, save_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Filter the data for the specified checkpoint step
    df_filtered = df[df['checkpoint_step'] == checkpoint_step]

    # Get the model size
    model_size = df['model_name'].iloc[0].split('-')[1]

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # Plot lines for each n_words_correlated value
    for n_words, group in df_filtered.groupby('n_words_correlated'):
        plt.plot(group['layer_num'], group['id'], marker='o', label=f'{n_words} words')

    plt.xlabel('Layer Number')
    plt.ylabel('Intrinsic Dimension')
    plt.title(f'Intrinsic Dimension over Layers (Checkpoint Step: {checkpoint_step})')
    plt.legend(title='Words Correlated')
    plt.tight_layout()

    # Save the plot
    plot_filename = f"intrinsic_dimension_over_layers_checkpoint_{checkpoint_step}_size_{model_size}.png"
    plt.savefig(f"{save_dir}/{plot_filename}", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved as {plot_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('--results_path', type=str, help='Path to results file')
    parser.add_argument('--save_dir', type=str, help='Directory to save plots')
    args = parser.parse_args()
    plot_id_diff_over_sizes(args.save_dir)
    plot_id_over_sizes(args.save_dir)
    plot_compositionality_gap_over_layers(args.results_path, args.save_dir)
    plot_intrinsic_dimension_over_layers(args.results_path, 143000, args.save_dir)
    plot_compositionality_over_time(args.results_path, args.save_dir)
# python plot.py --results_path /home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/pipeline_results_20240811_141329/EleutherAI_pythia-1.4b-dedupedresults.csv --save_dir /home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/pipeline_results_20240811_141329