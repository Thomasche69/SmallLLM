import os
import math
import pickle
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from Config import ModelConfig




class TextTokenDataset(Dataset):
    """
    Token dataset with configurable stride for creating training windows.

    Args:
        tokens: List of token IDs
        seq_len: Length of each sequence window
        stride: Step size between windows (default: seq_len for non-overlapping windows)
                - stride=seq_len: No overlap (most efficient)
                - stride=seq_len//2: 50% overlap
                - stride=1: Maximum overlap (1023x more samples, very inefficient)
    """
    def __init__(self, tokens: List[int], seq_len: int = 512, stride: int = None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len  # Default: non-overlapping

        # Calculate number of samples based on stride
        self.num_samples = max(0, (len(tokens) - seq_len) // self.stride + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Calculate actual token index based on stride
        start_idx = idx * self.stride
        x = torch.tensor(self.tokens[start_idx:start_idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[start_idx + 1:start_idx + self.seq_len + 1], dtype=torch.long)
        return x, y
    


def load_and_cache_data(config: ModelConfig, cache_dir: str = "data_cache"):
    """Load and cache tokenized data to avoid reprocessing"""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/tokenized_data_{config.num_documents}_{config.max_tokens}.pkl"

    # Check if cached data exists
    if os.path.exists(cache_file):
        print(f"📦 Loading cached data from {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        texts = cached_data['texts']
        tokenizer = cached_data['tokenizer']
        tokens = cached_data['tokens']
        config.vocab_size = tokenizer.vocab_size

        print(f"✅ Loaded {len(texts)} documents, {len(tokens):,} tokens from cache")
        return texts, tokenizer, tokens

    print(f"🔄 Processing new data (will cache for future use)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M", token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

    texts = []
    for i, item in enumerate(dataset):
        if i >= config.num_documents:
            break
        texts.append(item["text"])

    print(f"Loaded {len(texts)} documents")

    # Tokenize
    print("Tokenizing texts...")
    all_tokens = []
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

    tokens = all_tokens[:config.max_tokens]
    print(f"Using {len(tokens):,} tokens")
    config.vocab_size = tokenizer.vocab_size

    # Cache the processed data
    cached_data = {'texts': texts, 'tokenizer': tokenizer, 'tokens': tokens}
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)

    print(f"💾 Cached data to {cache_file}")
    return texts, tokenizer, tokens


def evaluate_model(model: nn.Module, val_loader: DataLoader, config: ModelConfig):
    """Evaluate model performance"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0

    device = next(model.parameters()).device

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= config.eval_steps:
                break
            x, y = x.to(device), y.to(device)

            with autocast():
                # MoE model evaluation
                logits = model(x, return_aux_losses=False)  # Don't return aux loss during eval
                loss = F.cross_entropy(logits.view(-1, config.vocab_size), y.view(-1))

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == y).sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    perplexity = math.exp(min(avg_loss, 20))

    model.train()
    return {'val_loss': avg_loss, 'val_accuracy': accuracy, 'val_perplexity': perplexity}

