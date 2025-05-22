import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from constants import *
import gc
import random

def generate_prompt(
    n_words_correlated, # number of words in sentence that should be grouped together
    shuffle=False,
):

  n_words_per_list = len(WORD_CATEGORIES[WORD_ORDER[0]])
  n_lists = len(WORD_ORDER)
  words = []
  word_idxes = np.zeros(n_lists)
  for i in range(n_lists):
    if i % n_words_correlated == 0:
      word_idx = np.random.randint(n_words_per_list)

    list_name = WORD_ORDER[i]
    words.append(WORD_CATEGORIES[list_name][word_idx])

  prompt = ""

  for i, word in enumerate(words):
    if i in DETERMINANTS:
      prompt += DETERMINANTS[i] + " "

    prompt += word + " "

  if shuffle:
    prompt = prompt[:-1].split(" ")
    random.shuffle(prompt)
    prompt = " ".join(prompt)
    prompt = prompt + '.'
  else:
    prompt = prompt[:-1] + "." # Replace final space


  return prompt

def generate_prompts(n, n_words_correlated, shuffle=False, rs=0):
  prompts = set()
  random.seed(rs)
  while len(prompts) < n:
    prompt = generate_prompt(n_words_correlated, shuffle)
    if prompt not in prompts:
        prompts.add(prompt)

  return list(prompts)



# Push representations through model
def encode_data(tokenizer, N, data, batch_size, max_length, device):
    # If the input data is text
    if type(data[0]) == str:
        encodings = tokenizer(data, padding=True, truncation=True, max_length=max_length, return_length=True, return_tensors="pt") # output variable length encodings
        encodings = [
            {'input_ids': encodings['input_ids'][i: i + batch_size].to(device),
            'attention_mask': encodings['attention_mask'][i: i + batch_size].to(device),
            'length': encodings['length'][i: i + batch_size] }
            for i in range(0, N, batch_size)
        ]
    else: # input data is tokens-- manually pad and batch.
        max_len = max([len(sentence) for sentence in data])
        data = [sentence for sentence in data if len(sentence) > 2]
        encodings = [tokenizer.encode(sentence[1:], padding='max_length', max_length=max_len, return_tensors="pt") \
                     for sentence in data]
        batched_encodings = [torch.stack(encodings[i: i + batch_size]).squeeze(1).to(device) for i in range(0, len(data), batch_size)]
        batched_attention_masks = [(tokens != 1).to(device).long() for tokens in batched_encodings]
        encodings = [
            {'input_ids': batched_encodings[j], 'attention_mask': batched_attention_masks[j]}
            for j in range(len(batched_encodings))
        ]

    return encodings

def last_token_rep(x, attention_mask, padding='right'):
    """
    Extracts the hidden representation of the last token in a sequence for a given layer (x).
    """
    seq_len = attention_mask.sum(dim=1)
    indices = (seq_len - 1)

    last_token_rep = x[torch.arange(x.size(0)), indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
    return last_token_rep.cpu()

def full_sequence(x, attention_mask, padding='right'):
    """
    Extracts the hidden representation of the last token in a sequence for a given layer (x).
    """
    seq_len = attention_mask.sum(dim=1)
    indices = (seq_len - 1)

    last_token_rep = x[torch.arange(x.size(0)), :indices] if padding=='right' else x[torch.arange(x.size(0)), -1]
    return last_token_rep.cpu()

def debug_memory():
    import collections, gc, resource, torch
    print('maxrss = {}'.format(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    tensors = collections.Counter(
        (str(o.device), str(o.dtype), tuple(o.shape))
        for o in gc.get_objects()
        if torch.is_tensor(o)
    )
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


def get_reps_from_llm(
    model,
    tokenizer,
    data,
    device,
    batch_size,
):
  # Tokenize the text data
  encodings = encode_data(tokenizer,
                          len(data),
                          data,
                          batch_size,
                          model.config.max_position_embeddings,
                          device
                          )

  with torch.no_grad():
    representations = []
    for batch in tqdm(encodings):
        output = model(batch['input_ids'], attention_mask=batch['attention_mask'], output_hidden_states=True)
        hiddens = output['hidden_states']
        pooled_output = tuple([last_token_rep(layer, batch['attention_mask'], padding=tokenizer.padding_side).cpu() for layer in hiddens])
        representations.append(pooled_output)

    # This step can be slow due to memory operations and data restructuring
    # Optimize by pre-allocating memory and using torch.stack instead of list comprehension and torch.cat
    num_layers = len(representations[0])
    num_samples = sum(batch[0].shape[0] for batch in representations)

    optimized_representations = []
    for layer_idx in range(num_layers):
        layer_tensor = torch.empty((num_samples, representations[0][layer_idx].shape[1]), dtype=representations[0][layer_idx].dtype, device='cpu')
        start_idx = 0
        for batch in representations:
            batch_size = batch[layer_idx].shape[0]
            layer_tensor[start_idx:start_idx+batch_size] = batch[layer_idx]
            start_idx += batch_size
        optimized_representations.append(layer_tensor)

    representations = optimized_representations

  return representations

def calculate_ids(
    representations,
    methods
):
    # For each layer, get nonlinear ID estimate
  IDS = {} # {'id method name' : list of ids over layers}
  representations = [rep.to(torch.float16) for rep in representations]

  # Compute ID
  for method in methods:
      IDS[method] = []
      print(f'computing ID for {method}')

      for layer_rep in tqdm(representations[1:]): # skip the positional embedding layer
          try:
            id = methods[method].fit_transform(layer_rep)
            IDS[method].append(id)
          except Exception as e:
            print(e)
            IDS[method].append(-1)

  return IDS

def get_model_and_tokenizer(model_name, model_step=None):
    # Load the model and tokenizer
    step = f"step{model_step}" if model_step is not None else None

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                )
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                load_in_8bit=True,
                                                )
    # Some idiosyncrasies of models
    if 'Llama' in model_name:
        tokenizer.padding_side = "right"
    if 'opt' not in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    if 'OLMo' in model_name:
        model.config.max_position_embeddings = model.config.max_sequence_length
    model.eval()

    return model, tokenizer

def run_pipeline(
    model,
    tokenizer,
    data,
    n_words_correlated,
    n_reps,
    methods,
    batch_size,
    device,
):
  with torch.no_grad():
    representations = get_reps_from_llm(model, tokenizer, data, device, batch_size)
    ids = calculate_ids(representations, methods)
    return ids

def load_results(results_dir):
    results = {}
    for model_name in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_name)
        results[model_name] = {}
        for checkpoint in os.listdir(model_dir):
            checkpoint_dir = os.path.join(model_dir, checkpoint)
            for file in os.listdir(checkpoint_dir):
                if file.endswith('.npz'):
                    file_path = os.path.join(checkpoint_dir, file)
                    with np.load(file_path, allow_pickle=True) as data:
                        n_words_correlated = int(data['n_words_correlated'])
                        checkpoint_step = int(data['checkpoint_step'])
                        for key in data['ids'].item().keys():
                            if checkpoint_step not in results[model_name]:
                                results[model_name][checkpoint_step] = {}
                            if n_words_correlated not in results[model_name][checkpoint_step]:
                                results[model_name][checkpoint_step][n_words_correlated] = {}
                            if key not in results[model_name][checkpoint_step][n_words_correlated]:
                                results[model_name][checkpoint_step][n_words_correlated][key] = {}

                            results[model_name][checkpoint_step][n_words_correlated][key] = data['ids'].item()[key]
    return results

def load_prompts(n_words_correlated, data_dir, shuffle=False, rs=None, sequence_length=None):
    rs_str = '' if rs is None else f"_rs{rs}"
    seq_str = '' if sequence_length is None else f'length_{sequence_length}_'
    with open(f'{data_dir}/train_prompts_{seq_str}{n_words_correlated}_words_correlated{rs_str}.txt', 'r') as f:
        train_prompts = f.read().splitlines()
    with open(f'{data_dir}/test_prompts_{seq_str}{n_words_correlated}_words_correlated{rs_str}.txt', 'r') as f:
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

