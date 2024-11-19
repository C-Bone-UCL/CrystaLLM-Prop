"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""

import os
import math
import time
import pickle
import gzip
import importlib
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
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

# Import utilities
from train_utils import (
    TrainDefaults,
    get_batch,
    is_model_structure_unmodified,
    estimate_loss,
    get_lr,
    objective,
    load_data_txt,
    load_data_table,
    initialize_model,
    implement_finetune_method,
    optimizer_setup,
    training_loop,
)

if __name__ == "__main__":
    # Parse the configuration
    C = parse_config(TrainDefaults)

    # CodeCarbon tracker
    if C.codecarbon:
        # make sure metrics_dir/wandb_project exists
        output_directory = f'{C.metrics_dir}/{C.wandb_project}'
        os.makedirs(output_directory, exist_ok=True)

        tracker = OfflineEmissionsTracker(
            output_dir=output_directory,
            project_name=f'{C.tracker_project}',
            country_iso_code="GBR",  # Replace with your country code
        )
        tracker.stop()
        tracker.start()

    # make the directory to store model ckpt if from scratch, or make sure it exists to resume
    os.makedirs(C.out_dir, exist_ok=True)
    if C.init_from == "scratch":
        print(f"Creating {C.out_dir}...")
    elif C.init_from == "resume":
        if not os.path.exists(C.out_dir):
            raise Exception(f"Could not find {C.out_dir} to resume from")
        else:
            print(f"Resuming from {C.out_dir}...")

    if C.hp_search:
        if not os.path.exists("./hp_search"):
            os.makedirs("./hp_search")

        # Optional: Set up logging for Optuna
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        # Create the Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"hyperparameter_search_{C.adaptation}_{C.finetune_method}",
            load_if_exists=True,
            storage=f"sqlite:///./hp_search/{C.adaptation}_{C.finetune_method}_test.db",
            pruner=optuna.pruners.MedianPruner(),
        )

        # Run the hyperparameter optimization
        study.optimize(lambda trial: objective(trial, C), n_trials=C.n_trials_hp_search)

        # Retrieve and print the best hyperparameters and validation loss
        best_params = study.best_params
        best_value = study.best_value

        print("Best Hyperparameters:", best_params)
        print("Best Validation Loss:", best_value)

    else:
        # set the random seed, device and data type
        torch.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
        # note: float16 data type will automatically use a GradScaler
        ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
        ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # Load the dataset
        if not C.dataset:
            raise ValueError("The 'dataset' option is required and cannot be empty")
        if C.adaptation == 'base' or C.adaptation == 'cifextd':
            train_data, val_data, cif_start_indices, cif_start_indices_val, cif_start_indices_underrep, tokenizer, vocab_size, unk_token_id = load_data_txt(C)
            max_token_length = None
        elif C.adaptation == 'regression':
            tokenizer, vocab_size, train_loader, val_loader, test_loader, unk_token_id, max_token_length = load_data_table(C)
        else:
            raise ValueError(f"Unknown adaptation type: {C.adaptation}")

        model, iter_num, best_val_loss, model_args, checkpoint = initialize_model(C, vocab_size, unk_token_id, max_token_length)

        # move the model to the correct device
        model.to(C.device)
        # initialize a GradScaler; if enabled=False scaler is a no-op
        scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))
        # Implement the finetuning method
        model = implement_finetune_method(C, model, checkpoint)

        # Setup the optimizer
        optimizer, model = optimizer_setup(C, model, checkpoint)

        # helps estimate an arbitrarily accurate loss over either split using many batches
        loader_iter = {}
        # get_batch_func is a partial function with necessary arguments
        if C.adaptation == 'base' or C.adaptation == 'cifextd':
            get_batch_func = lambda split: get_batch(
                split, C, device_type, train_data, val_data, cif_start_indices, cif_start_indices_val, cif_start_indices_underrep, loader_iter=loader_iter
            )
        elif C.adaptation == 'regression':            
            get_batch_func = lambda split: get_batch(
                split, C, device_type, train_loader=train_loader, val_loader=val_loader, loader_iter=loader_iter
            )

        # training loop
        training_loop(model, model_args, optimizer, scaler, get_batch_func, C, ctx, device_type, iter_num, best_val_loss, trial=None)

    # Finalize tracker
    if C.codecarbon:
        tracker.stop()
