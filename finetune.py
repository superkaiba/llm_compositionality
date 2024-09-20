from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import TrainingArguments, Trainer
import torch
from datasets import load_dataset
import argparse

from finetuning_callbacks import *

# Initialize the parser
parser = argparse.ArgumentParser(description="Finetuning script")

# Add arguments
parser.add_argument('--dataset', type=str, help='correlated words 1 2 3 4', choices=[1, 2, 3, 4, 'pile'])
parser.add_argument('--model_name', type=str, default='EleutherAI/pythia-410m-deduped')
parser.add_argument('--debug', type=int, default=0, help='1 if true')

args = parser.parse_args()
print(args)

torch.manual_seed(42)
MODEL = args.model_name
DATASET = args.dataset

# LOAD DATA (train on one random split, eventually evaluate on many)
train_fpath = f'/home/echeng/llm_compositionality/data/train_prompts_{args.dataset}_words_correlated_rs0.txt'
val_fpath = f'/home/echeng/llm_compositionality/data/test_prompts_{args.dataset}_words_correlated_rs0.txt'

dataset = load_dataset('text', data_files={'train': train_fpath, 'val': val_fpath})

# LOAD MODEL
model = AutoModelForCausalLM.from_pretrained(
        MODEL,
)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=1, remove_columns=['text'])
train_dataset = tokenized_dataset['train']
val_dataset = tokenized_dataset['val']

def group_texts(examples):
    # Concatenate all texts into one long string of token IDs
    concatenated = {k: list(sum(examples[k], [])) for k in examples.keys()}

    # Calculate total length and ensure it's a multiple of block_size
    total_length = len(concatenated[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split the concatenated data into chunks of block_size
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()

    }

    # Copy input_ids to labels for causal language modeling
    result["labels"] = result["input_ids"].copy()

    return result

# Example block size (set according to model's requirement
block_size = 20

# Apply the grouping function to the tokenized dataset
lm_train_dataset = train_dataset.map(group_texts, batched=True)
lm_val_dataset = val_dataset.map(group_texts, batched=True)
lm_train_dataset.set_format("torch")
lm_val_dataset.set_format("torch")

# FINETUNE
BATCH_SIZE = 1
N_EPOCHS = 10
total_steps = len(lm_train_dataset) // BATCH_SIZE * N_EPOCHS

training_args = TrainingArguments(
        f"emcheng/{MODEL}-finetuned-{DATASET}-words-correlated",
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        num_train_epochs=N_EPOCHS, 
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        # save_steps=0.5,
        save_strategy='epoch',
        # save_only_model=True,
        report_to=[],
        fp16=True,  # Use mixed precision training if supported by your hardware
        
)


trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_train_dataset,
        eval_dataset=lm_val_dataset,
        callbacks=[SaveAtFractionalEpochCallback(save_fractions=[x / N_EPOCHS for x in [0.125, 0.25, 0.5, 0.75]], 
                                             num_train_epochs=N_EPOCHS,
                                             total_steps=total_steps)]
)

trainer.train()


