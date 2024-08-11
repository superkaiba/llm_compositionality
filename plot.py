from utils import load_results
import matplotlib.pyplot as plt
import pandas as pd
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
    df = pd.read_csv(csv_file)
    layer_numbers = df['layer_num'].unique()
    fig, axes = plt.subplots(6, 4)

    for layer_number in layer_numbers:
        ax = axes[layer_number // 4, layer_number % 4]
        layer_df = df[df['layer_num'] == layer_number]
        n_words_correlated = layer_df['n_words_correlated'].unique()
        for n_words in n_words_correlated:
            n_words_correlated_df = layer_df[layer_df['n_words_correlated'] == n_words]
            ax.plot(n_words_correlated_df['checkpoint_step'], n_words_correlated_df['id'], label=f"{n_words}")

        ax.set_xlabel('Training epoch')
        ax.set_ylabel('ID')
        # ax.legend()
        ax.set_title(f"Layer {layer_number}")
        handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', title='Number of correlated words', ncol=4)
    fig.tight_layout()
    fig.savefig(f"{save_dir}/compositionality_over_time.png")

plot_compositionality_over_time("/home/mila/t/thomas.jiralerspong/llm_compositionality/results/pipeline_results_20240810_171441/EleutherAI_pythia-410m-dedupedresults.csv", "/home/mila/t/thomas.jiralerspong/llm_compositionality/results/pipeline_results_20240810_171441")