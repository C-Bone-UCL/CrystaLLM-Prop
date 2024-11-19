import os
import math
import time
import pickle
import gzip
import importlib
from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch._dynamo
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

from codecarbon import OfflineEmissionsTracker
import optuna
import sys
import wandb

from crystallm import (
    parse_config,
    CIFTokenizer,
    CIFTokenizer_extd,
    GPT,
    GPTConfig,
    GPT_regression,
    _model,
    _tokenizer_cifextd,
    _tokenizer,
)

# For PyTorch
torch.cuda.empty_cache()
torch._dynamo.config.verbose = True

importlib.reload(_model)
importlib.reload(_tokenizer_cifextd)
importlib.reload(_tokenizer)

@dataclass
class TrainDefaults:
    out_dir: str = "out" # the path to the folder where the model checkpoints will be stored
    ckpt_out_dir: str = "out" # the path to the folder where the model checkpoints will be stored
    eval_interval: int = 250 # how often to evaluate against the validation set
    log_interval: int = 1 # how often to print log messages
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False # if True, script exits right after the first eval
    always_save_checkpoint: bool = False # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # Options: 'scratch' or 'resume'

    # WandB logging
    wandb_log: bool = False
    wandb_project: str = 'crystallm_BG_CIF'
    wandb_run_name: str = 'BG_large'

    # Data parameters
    dataset: str = "" # the path to the dataset
    gradient_accumulation_steps: int = 40 # concatenate this many batches before backward/update
    batch_size: int = 64
    block_size: int = 2048 # context of up to `block_size` previous character

    # Model parameters
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0 # prevent overfitting, useful for finetune mainly
    bias: bool = False

    # Optimizer parameters
    learning_rate: float = 6e-4
    max_iters: int = 600000 # total number of iterations for training
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate decay settings
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # System settings
    device: str = "cuda" # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16" # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True
    underrep_p: float = 0.0
    validate: bool = False # whether to evaluate the model using the validation set

    # Logging metrics
    codecarbon: bool = False # if True, log emissions to CodeCarbon
    tracker_project: str = "crystallm"
    metrics_dir: str = "comp_metrics/"

    # LoRA parameters
    LoRA_rank: int = 16
    LoRA_alpha: int = 32

    # Finetuning method
    finetune_method: str = 'finetune_all'  # Options: 'finetune_head', 'finetune_all', 'LoRA'
    adaptation: str = "base" # Options: 'base', 'regression', 'cifextd'

    # Sanity check
    sanity_check: bool = False # if True, print parameter changes after update

    # Train-test split for regression adaptation
    test_size: float = 0.05
    val_size: float = 0.05

    # hp search
    hp_search: bool = False
    n_trials_hp_search: int = 1
    
# Load configuration
C = parse_config(TrainDefaults)

def objective(trial):
    """
    Objective function for the hyperparameter search.
    Returns the validation loss for the given hyperparameters.
    """

    # Suggest values for the hyperparameters
    n_layer = trial.suggest_int('n_layer', 4, 12)
    n_head = trial.suggest_int('n_head', 4, 12)
    n_embd = trial.suggest_int('n_embd', 256, 1024, step=64)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

    # Update the configuration with the suggested hyperparameters
    C.n_layer = n_layer
    C.n_head = n_head
    C.n_embd = n_embd
    C.learning_rate = learning_rate
    C.weight_decay = weight_decay
    C.dropout = dropout

    # Initialize model arguments
    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head,
        n_embd=C.n_embd,
        block_size=C.block_size,
        bias=C.bias,
        vocab_size=vocab_size,
        dropout=C.dropout,
    )

    # Initialize the model
    gptconf = GPTConfig(**model_args)
    if C.adaptation == 'base' or C.adaptation == 'cifextd':
        model = GPT(gptconf)
    elif C.adaptation == 'regression':
        model = GPT_regression(gptconf)
    else:
        model = None  # Handle other cases if necessary

    model.to(C.device)

    # Initialize the optimizer with trial's learning rate and weight decay
    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))

    # Initialize the GradScaler
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    # Training loop (you can limit the number of iterations for the hyperparameter search)
    max_iters = 1000  # Or any suitable number
    iter_num = 0
    best_val_loss = float('inf')

    while iter_num < max_iters:
        # Training step
        X, Y = get_batch("train")
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # Backward pass
            scaler.scale(loss).backward()

        # Gradient clipping
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Evaluate on validation set periodically
        if iter_num % C.eval_interval == 0:
            losses = estimate_loss()
            val_loss = losses['val']
            trial.report(val_loss, iter_num)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.TrialPruned()

        iter_num += 1

    return best_val_loss