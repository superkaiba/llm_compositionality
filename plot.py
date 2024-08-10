from utils import load_results
import matplotlib.pyplot as plt
results = load_results("/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/pipeline_results_20240810_135811/EleutherAI")
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
    
plot_results(results, "/home/mila/t/thomas.jiralerspong/llm_compositionality/scratch/llm_compositionality/results/pipeline_results_20240810_135811/EleutherAI")



