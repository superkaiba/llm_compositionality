import random
import matplotlib.pyplot as plt

def load_prompts(n_words_correlated, data_dir, shuffle=False):
    with open(f'{data_dir}/train_prompts_{n_words_correlated}_words_correlated.txt', 'r') as f:
        train_prompts = f.read().splitlines()
    with open(f'{data_dir}/test_prompts_{n_words_correlated}_words_correlated.txt', 'r') as f:
        test_prompts = f.read().splitlines()
    if shuffle:
        for i in range(len(train_prompts)):
            train_prompts[i] = train_prompts[i].split(" ")
            random.shuffle(train_prompts[i])
            train_prompts[i] = " ".join(train_prompts[i])
        for i in range(len(test_prompts)):
            test_prompts[i] = test_prompts[i].split(" ")
            random.shuffle(test_prompts[i])
            test_prompts[i] = " ".join(test_prompts[i])

    
    return train_prompts, test_prompts


def compute_unigram_frequency(data_dir):
    unigram_freq = {}
    all_words = set()
    total_word_count = {}
    
    fig, axs = plt.subplots(2, 4, figsize=(30, 15))
    fig.suptitle('Unigram Frequencies (All Words)', fontsize=16)
    
    # First pass: collect all unique words and total counts
    for n_words_correlated in range(1, 5):
        for shuffle in [False, True]:
            train_prompts, test_prompts = load_prompts(n_words_correlated, data_dir, shuffle)
            all_prompts = train_prompts + test_prompts
            
            for prompt in all_prompts:
                words = prompt.lower().split()
                for word in words:
                    if word not in ['the', 'then']:
                        all_words.add(word)
                        total_word_count[word] = total_word_count.get(word, 0) + 1
    
    # Sort words by total frequency across all combinations
    all_words = sorted(list(all_words), key=lambda w: total_word_count[w], reverse=True)
    
    # Print top 50 words
    print("Top 50 words:")
    for i, word in enumerate(all_words[:50], 1):
        print(f"{i}. {word}: {total_word_count[word]}")
    
    # Second pass: compute frequencies and plot
    for n_words_correlated in range(1, 5):
        for shuffle in [False, True]:
            train_prompts, test_prompts = load_prompts(n_words_correlated, data_dir, shuffle)
            all_prompts = train_prompts + test_prompts
            
            word_count = {word: 0 for word in all_words}
            total_words = 0
            
            for prompt in all_prompts:
                words = prompt.lower().split()
                total_words += len([w for w in words if w not in ['the', 'then']])
                for word in words:
                    if word not in ['the', 'then']:
                        word_count[word] += 1
            
            # Calculate frequency
            freq = {word: count / total_words for word, count in word_count.items()}
            
            key = f"{n_words_correlated}_words_{'shuffled' if shuffle else 'unshuffled'}"
            unigram_freq[key] = freq
            
            # Plot frequencies for all words
            ax = axs[int(shuffle), n_words_correlated-1]
            frequencies = [freq[word] for word in all_words]
            ax.bar(range(len(all_words)), frequencies)
            ax.set_title(f"{n_words_correlated} words {'shuffled' if shuffle else 'unshuffled'}")
            ax.set_xlabel('All Words')
            ax.set_ylabel('Frequency')
            ax.set_xticks([])  # Remove x-axis ticks as there are too many words
    
    # Add row labels
    fig.text(0.08, 0.75, 'Unshuffled', rotation=90, fontsize=14, va='center')
    fig.text(0.08, 0.25, 'Shuffled', rotation=90, fontsize=14, va='center')
    
    # Add column labels
    for i, n_words in enumerate(range(1, 5)):
        fig.text((i + 0.5) / 4, 0.95, f'{n_words} words correlated', ha='center', fontsize=14)
    
    plt.tight_layout(rect=[0.1, 0.03, 1, 0.95])
    plt.savefig(f'{data_dir}/unigram_frequencies_all_words.png')
    plt.close()
    
    return unigram_freq

def plot_max_unigram_difference(data_dir):
    # First, compute the unigram frequencies
    unigram_freq = compute_unigram_frequency(data_dir)
    
    # Get all unique words across all combinations
    all_words = set()
    for freq in unigram_freq.values():
        all_words.update(freq.keys())
    
    # Calculate max difference for each word
    max_differences = {}
    for word in all_words:
        counts = [freq.get(word, 0) * total_words for freq, total_words in zip(unigram_freq.values(), [sum(freq.values()) for freq in unigram_freq.values()])]
        max_diff = max(counts) - min(counts)
        max_differences[word] = max_diff
    
    # Sort words by max difference
    sorted_words = sorted(max_differences.items(), key=lambda x: x[1], reverse=True)
    
    # Plot
    plt.figure(figsize=(15, 8))
    words, differences = zip(*sorted_words)
    plt.bar(range(len(words)), differences)
    plt.title('Maximum Difference in Unigram Counts Across All Combinations')
    plt.xlabel('Words (sorted by max difference)')
    plt.ylabel('Maximum Count Difference')
    plt.xticks([])  # Remove x-axis ticks as there are too many words
    
    # Add labels for top N words
    N = 20  # Number of top words to label
    for i in range(N):
        plt.text(i, differences[i], words[i], ha='center', va='bottom', rotation=90)
    
    plt.tight_layout()
    plt.savefig(f'{data_dir}/max_unigram_count_difference.png')
    plt.close()

# Example usage:
# plot_max_unigram_difference('./')



compute_unigram_frequency('/home/mila/t/thomas.jiralerspong/llm_compositionality/data')
# Example usage:
# unigram_frequencies = compute_unigram_frequency('path/to/data/directory')

