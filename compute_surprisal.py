from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import argparse
import json
from tqdm import tqdm
import pdb

parser = argparse.ArgumentParser(description='surprisal computation')
# Data selection
parser.add_argument('--model_name', type=str, default="EleutherAI/pythia-410m-deduped")
parser.add_argument('--dataset', type=int, choices=[1, 2, 3, 4])
parser.add_argument('--epoch', type=float, choices=[0, 0.125, 0.25, 1, 2, 3, 4]) # 0 = pretrained model
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()
print(args)

epoch_to_ckpt = {0:0, 0.125: 25600, 0.25: 51200, 1:153600, 2:307200, 3:460800, 4:614400}
ckpt = epoch_to_ckpt[args.epoch]
if ckpt == 0:
    model_path = args.model_name
else:
    model_path = f'/home/echeng/llm_compositionality/emcheng/{args.model_name}-finetuned-{args.dataset}-words-correlated/checkpoint-{ckpt}'
train_dataset_path = f'/home/echeng/llm_compositionality/data/train_prompts_{args.dataset}_words_correlated.txt'
test_dataset_path = f'/home/echeng/llm_compositionality/data/test_prompts_{args.dataset}_words_correlated.txt'

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                          trust_remote_code=True,
                                          )
# pdb.set_trace()
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_8bit=True,
                                            )

tokenizer.pad_token = tokenizer.eos_token

model.eval()
print('model loaded')

# Load dataset
with open(train_dataset_path, 'r') as f:
    train_dataset = [l[:-1] for l in f.readlines()]

with open(test_dataset_path, 'r') as f:
    test_dataset = [l[:-1] for l in f.readlines()]

# Calculate surprisal
def calculate_surprisal(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(model.device)

    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits

    # Shift logits and input_ids to align them for calculating next-token probabilities
    shifted_logits = logits[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]

    # Calculate the log probabilities of the true next token
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

    # Surprisal is the negative log probability
    sum_token_surprisal = -float(token_log_probs.cpu().sum().item())
    num_tokens = token_log_probs.shape[1]

    return sum_token_surprisal, num_tokens

# Calculate surprisal for each text in the dataset
train_surprisals = [calculate_surprisal(text) for text in tqdm(train_dataset) if len(text) > 0]
train_surp, train_lens = [result[0] for result in train_surprisals], [result[1] for result in train_surprisals]
avg_train_surprisal = sum(train_surp) / sum(train_lens)
train_surp = [result[0] / result[1] for result in train_surprisals]

test_surprisals = [calculate_surprisal(text) for text in tqdm(test_dataset) if len(text) > 0]
test_surp, test_lens = [result[0] for result in test_surprisals], [result[1] for result in test_surprisals]
avg_test_surprisal = sum(test_surp) / sum(test_lens)
test_surp = [result[0] / result[1] for result in test_surprisals]

# Save results, guarantee is it's the same order as the txt files
results = {'per_token_train_surprisal_sequence': train_surp,
           'train_len': train_lens, 
           'train_prompts': train_dataset, 
           'per_token_test_surprisal_sequence': test_surp,
           'test_len': test_lens, 
           'test_prompts': test_dataset,
           'avg_per_token_train_surprisal_corpus': avg_train_surprisal,
           'avg_per_token_test_surprisal_corpus': avg_test_surprisal
           }

with open(f'/home/echeng/llm_compositionality/data/surprisals/{args.model_name.replace("/", "_")}_step{ckpt}_{args.dataset}_words_correlated.json', 'w') as f:
    json.dump(results, f)