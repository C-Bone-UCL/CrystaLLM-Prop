import os
import math
import time
import pickle
import gzip
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from contextlib import nullcontext
import logging
import optuna
import sys
import wandb
from copy import deepcopy
import random

from omegaconf import OmegaConf

from codecarbon import OfflineEmissionsTracker

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

import importlib
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

    # latent dimension for regression adaptation
    latent_dim: int = 256

def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> torch.Tensor:
    start_indices = None
    starts_path = os.path.join(data_dir, starts_fname)
    if on_condition:
        if os.path.exists(starts_path):
            print(f"Reading start indices from {starts_path}...")
            with open(starts_path, "rb") as f:
                start_indices = torch.tensor(pickle.load(f))  # should be sorted
            # Remove indices that would result in out-of-bounds sequences
            start_indices = start_indices[start_indices <= max_start_index]
        elif required:
            raise FileNotFoundError(f"Expected to find a file in dataset dir named '{starts_fname}'")
    return start_indices

class CIFRegressionDataset(Dataset):
    def __init__(self, dataframe, max_length, unk_token_id):
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.unk_token_id = unk_token_id  # ID for the <unk> token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.loc[idx, 'CIFs_tokenized']

        # Efficiently pad or truncate using numpy
        x = np.full(self.max_length, self.unk_token_id, dtype=np.int64)  # Initialize with <unk> token ID
        token_ids = [self.unk_token_id if token == '<unk>' else int(token) for token in tokens]
        x[:min(len(token_ids), self.max_length)] = token_ids[:self.max_length]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(self.data.loc[idx, 'Bandgap (eV)'], dtype=torch.float32)
        return x, y

def get_batch(
    split, C, device_type, train_data=None, val_data=None, cif_start_indices=None, 
    cif_start_indices_val=None, cif_start_indices_underrep=None, train_loader=None, 
    val_loader=None, loader_iter=None
):
    if C.adaptation == 'base' or C.adaptation == 'cifextd':
        data = train_data if split == "train" else val_data

        ix = torch.randint(len(data) - C.block_size, (C.batch_size,))
        if split == "train":
            if C.underrep_p > 0 and np.random.rand() < C.underrep_p:
                ix = cif_start_indices_underrep[torch.randperm(len(cif_start_indices_underrep))[:C.batch_size]]
            elif cif_start_indices is not None:
                ix = cif_start_indices[torch.randperm(len(cif_start_indices))[:C.batch_size]]
        elif cif_start_indices_val is not None:
            ix = cif_start_indices_val[torch.randperm(len(cif_start_indices_val))[:C.batch_size]]

        x = torch.stack([torch.from_numpy(data[i:i + C.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + C.block_size].astype(np.int64)) for i in ix])

    elif C.adaptation == 'regression':
        loader = train_loader if split == 'train' else val_loader
        # Check if the iterator for the split exists; if not, create it
        if split not in loader_iter or loader_iter[split] is None:
            loader_iter[split] = iter(loader)
        try:
            x, y = next(loader_iter[split])
        except StopIteration:
            # Reinitialize the iterator if it is exhausted
            loader_iter[split] = iter(loader)
            x, y = next(loader_iter[split])

    if device_type == "cuda":
        x = x.pin_memory().to(C.device, non_blocking=True)
        y = y.pin_memory().to(C.device, non_blocking=True)
    else:
        x = x.to(C.device)
        y = y.to(C.device)

    return x, y

def load_data_txt(C):
    """
    Load the dataset based on the adaptation method.
    Returns necessary variables for training.
    """
    # Load binary data files for training and validation
    train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None

    # Read start indices for sampling sequences
    cif_start_indices = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts.pkl",
    )

    cif_start_indices_val = read_start_indices(
        max_start_index=(len(val_data) - C.block_size) if C.validate else -1,
        data_dir=C.dataset,
        starts_fname="starts_val.pkl",
        on_condition=C.validate,
    )

    cif_start_indices_underrep = read_start_indices(
        max_start_index=len(train_data) - C.block_size,
        data_dir=C.dataset,
        starts_fname="starts_underrep.pkl",
        on_condition=C.underrep_p > 0,
        required=True,
    )

    # Initialize tokenizer and vocab size
    if C.adaptation == "base":
        tokenizer = CIFTokenizer()
        vocab_size = len(tokenizer._tokens_with_unk)
        unk_token_id = tokenizer.token_to_id["<unk>"]
    elif C.adaptation == "cifextd":
        tokenizer = CIFTokenizer_extd()
        vocab_size = len(tokenizer._tokens_with_unk)
        unk_token_id = tokenizer.token_to_id["<unk>"]

    return (
        train_data,
        val_data,
        cif_start_indices,
        cif_start_indices_val,
        cif_start_indices_underrep,
        tokenizer,
        vocab_size,
        unk_token_id,
    )

def load_data_table(C):
    # Load the DataFrame from the compressed file
    with gzip.open(C.dataset, 'rb') as f:
        df = pickle.load(f)

    # Calculate the maximum token length using pandas
    max_token_length = df['CIFs_tokenized'].str.len().max()

    # Split data into train, validation, and test sets
    train_df, test_df = train_test_split(df, test_size=C.test_size, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=C.val_size / (1 - C.test_size), random_state=42)

    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}\n")

    # Initialize tokenizer and get the <unk> token ID
    tokenizer = CIFTokenizer()
    unk_token_id = tokenizer.token_to_id["<unk>"]

    # Create regression datasets
    train_dataset = CIFRegressionDataset(train_df, max_token_length, unk_token_id)
    val_dataset = CIFRegressionDataset(val_df, max_token_length, unk_token_id)
    test_dataset = CIFRegressionDataset(test_df, max_token_length, unk_token_id)

    # Save the test dataset for future use
    test_out_dir = os.path.dirname(C.dataset)
    test_out_path = os.path.join(test_out_dir, "test_dataset.pkl.gz")
    with gzip.open(test_out_path, 'wb') as f:
        pickle.dump(test_df, f)
    print(f"Saved test dataset to {test_out_path}\n")

    # Save the test dataset for future use
    val_out_dir = os.path.dirname(C.dataset)
    val_out_path = os.path.join(val_out_dir, "val_dataset.pkl.gz")
    with gzip.open(val_out_path, 'wb') as f:
        pickle.dump(val_df, f)
    print(f"Saved val dataset to {val_out_path}\n")

    # Get vocab size from tokenizer
    vocab_size = len(tokenizer._tokens_with_unk)

    # Create DataLoaders for regression
    train_loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=C.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=C.batch_size, shuffle=False)

    return (
        tokenizer,
        vocab_size,
        train_loader,
        val_loader,
        test_loader,
        unk_token_id,
        max_token_length
    )

def is_model_structure_unmodified(model, checkpoint):
    """
    Compares the current model's parameter names and shapes to the checkpoint.
    Returns True if they match (unmodified structure), False if not.
    """
    checkpoint_state = checkpoint["model"]  # Access checkpoint state dictionary

    # Get current model's state dictionary
    model_state = model.state_dict()

    # Compare parameter names and shapes
    for param_name, param_tensor in model_state.items():
        if param_name not in checkpoint_state:
            print(f"Parameter {param_name} missing in checkpoint.\n")
            return False  # Parameter was added
        if param_tensor.shape != checkpoint_state[param_name].shape:
            print(f"Parameter {param_name} shape mismatch: {param_tensor.shape} vs {checkpoint_state[param_name].shape}\n")
            return False  # Parameter shape has changed

    # Check for any extra parameters in the checkpoint that aren't in the model
    for param_name in checkpoint_state:
        if param_name not in model_state:
            print(f"Checkpoint parameter {param_name} missing in current model.\n")
            return False  # Parameter was removed

    # If all parameters and shapes match
    return True

@torch.no_grad()
def estimate_loss(C, model, ctx, get_batch_func, device_type):
    out = {}
    model.eval()
    for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_func(split)  # Only pass 'split' here
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it, C):
    # 1) linear warmup for warmup_iters steps
    if it < C.warmup_iters:
        return C.learning_rate * it / C.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > C.lr_decay_iters:
        return C.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - C.warmup_iters) / (C.lr_decay_iters - C.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return C.min_lr + coeff * (C.learning_rate - C.min_lr)


def initialize_model(C, vocab_size, unk_token_id, max_token_length=None):
    # Initialize the model arguments
    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head,
        n_embd=C.n_embd,
        block_size=C.block_size,
        bias=C.bias,
        vocab_size=vocab_size,
        dropout=C.dropout,
    )

    if C.init_from == "scratch":
        # Initialize a new model from scratch with the given model arguments
        print("Initializing a new model from scratch...\n")
        gptconf = GPTConfig(**model_args)
        if C.adaptation in ['base', 'cifextd']:
            model = GPT(gptconf)
        elif C.adaptation == 'regression':
            model = GPT_regression(gptconf)
        else:
            model = None
        iter_num = 0
        best_val_loss = 1e-9

    elif C.init_from == "resume":
        # Update model arguments to load so they match checkpoint
        print(f"Resuming training from {C.out_dir}...\n")
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        model_args.update(checkpoint_model_args)
        model_args["finetune_method"] = C.finetune_method
        model_args["sanity_check"] = C.sanity_check
        model_args["unk_token_id"] = unk_token_id
        model_args["max_token_length"] = checkpoint_model_args.get("block_size")
        model_args["latent_dim"] = C.latent_dim
        print("Using configuration:")
        print(OmegaConf.to_yaml(C))

        # Initialize the model with model arguments from whatever checkpoint is loaded (pretrained, checkpoint, etc.)
        # We do this because if we load the model but change arguments to what we want to finetune with
        # then the model will not have the correct parameters and error out
        # so load model with checkpoint args and then update the args
        gptconf = GPTConfig(**model_args)
        if C.adaptation in ['base', 'cifextd']:
            model = GPT(gptconf)
        elif C.adaptation == 'regression':
            model = GPT_regression(gptconf)
        else:
            model = None

        # Load state_dict and adjust as needed then remove unwanted prefixes
        state_dict = checkpoint["model"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        if C.adaptation == 'regression':
            # Remove outdated 'lm_head' keys
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head")}

        # Load the state_dict with strict=False to ignore missing keys (necessary for regression adaptation)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model successfully from {ckpt_path}\n")

        # Check for vocabulary size mismatch and resize if necessary (implemented for base or cifextd adaptation)
        checkpoint_vocab_size = checkpoint_model_args.get("vocab_size")
        if checkpoint_vocab_size < vocab_size:
            print(f"Vocabulary size mismatch: checkpoint for finetuning has {checkpoint_vocab_size}, tokenizer initialised has {vocab_size}\n")
            model.resize_token_embeddings(vocab_size)
            model.transformer.wte.weight.grad = None  # Clear gradients for resized embeddings
            # do I need to also clear the head weights?
            model_args["vocab_size"] = vocab_size  # Update model_args
            print(f"Resized model token embeddings to {vocab_size}")
        elif checkpoint_vocab_size > vocab_size:
            print(f"Vocabulary size mismatch: checkpoint for finetuning has {checkpoint_vocab_size}, tokenizer initialised has {vocab_size}\n")
            raise ValueError("The new dataset has a smaller vocab size than the checkpoint which causes tensor errors\n")

        # Replace linear layers with LoRA layers if this is the finetuning method
        if C.finetune_method == 'LoRA':
            print("Replacing linear layers with LoRA layers\n")
            model.replace_linear_with_lora(rank=C.LoRA_rank, alpha=C.LoRA_alpha)

        # Load best val loss from pretrained model if adaptation isnt "base"
        if C.adaptation != 'base':
            os.makedirs(C.ckpt_out_dir, exist_ok=True)
            ckpt_out_path = os.path.join(C.ckpt_out_dir, "ckpt.pt")
            if os.path.exists(ckpt_out_path):
                checkpoint_out = torch.load(ckpt_out_path, map_location=C.device)
                best_val_loss = checkpoint_out.get("best_val_loss", 1e9)
                print(f'Best val loss from checkpoint: {best_val_loss}\n')
            else:
                best_val_loss = 1e9
                print(f'No checkpoint found at {ckpt_out_path}, so best_val_loss_out is set to {best_val_loss}\n')
            iter_num = 0
        # Load the best val loss from the checkpoint if it exists to resume a "base" training
        # Or set to 1e9 if from scratch
        elif C.adaptation == 'base':
            best_val_loss = checkpoint.get("best_val_loss", 1e9)
            print(f'Best val loss from checkpoint: {best_val_loss}\n')
            iter_num = 0
    else:
        raise ValueError(f"Unknown init_from value: {C.init_from}\n")

    # Crop down the model block size if necessary
    # When handling next token generation, if we are loading a checkpoint we dont want to have a larger block size
    # than the model was trained on, no point in having a larger block size than the model was trained on
    # as it will just be padding
    if C.adaptation in ['base', 'cifextd']:
        if C.block_size < model.config.block_size:
            model.crop_block_size(C.block_size)
            model_args["block_size"] = C.block_size  # Update model_args
            print(f"Cropped model block from the loaded ckpt to {C.block_size}\n")
    # for regression adaptation, we need to take in however many tokens are in the longest X input (longest CIF)
    # so that we can make Y predictions from whole CIF files, not just up to the block size
    # so the max_token_length is amt of tokens in longest CIF file in the dataset
    if C.adaptation == 'regression':
        if max_token_length != model.config.block_size:
            model_args["max_token_length"] = max_token_length
            model.resize_block(max_token_length)
            model_args["max_token_length"] = max_token_length

    return model, iter_num, best_val_loss, model_args, checkpoint

def implement_finetune_method(C, model, checkpoint):
    # Finetuning methods
    # Just telling which layers to train and which to freeze for each finetune method
    if C.init_from == "resume" and C.finetune_method == 'finetune_head':
        print("Finetuning head only: Freezing transformer layers\n")
        for param in model.transformer.parameters():
            param.requires_grad = False
        model.transformer.wte.weight.requires_grad = True
    elif C.init_from == "resume" and C.finetune_method == 'finetune_all':
        print("Finetuning all layers\n")
        for param in model.parameters():
            param.requires_grad = True
    elif C.init_from == "resume" and C.finetune_method == 'LoRA':
        print("Applying LoRA finetuning\n")
        model.replace_linear_with_lora(rank=C.LoRA_rank, alpha=C.LoRA_alpha)
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model

def optimizer_setup(C, model, checkpoint):
    # Optimizer and GradScaler setup
    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))

    # Load optimizer state from checkpoint if model structure is unmodified, otherwise reset optimizer
    if C.init_from == "resume":
        model_structure_unmodified = is_model_structure_unmodified(model, checkpoint)
        if model_structure_unmodified:
            print("Loading optimizer state from checkpoint...\n")
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Model structure modified; resetting optimizer.\n")
    else:
        print("Initializing optimizer from scratch.\n")

    # Compile the model if desired
    if C.compile:
        print("Compiling the model (takes a ~minute)...\n")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0
        model.to(C.device)

    # If ft method is LoRA, freeze all layers except LoRA layers
    if C.init_from == "resume" and C.finetune_method == 'LoRA':
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Print a summary of trainable versus frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({100.0 * trainable_params / total_params:.2f}%)\n")
    if C.sanity_check:
        # Sanity Check: Ensure that the correct parameters are set as trainable
        print("===== Sanity Check: Trainable Parameters =====")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
        if C.adaptation == 'regression':
            for name, param in model.lm_head.named_parameters():
                if 'lm_head' in name:
                    if param.requires_grad:
                        print(f"Trainable: {name}")
                    else:
                        print(f"Frozen: {name}")
        elif C.adaptation == 'base' or C.adaptation == 'cifextd' and C.finetune_method != 'LoRA':
            if model.lm_head.weight.requires_grad:
                print("Trainable: _orig_mod.lm_head.weight")
            else:
                print("Frozen: _orig_mod.lm_head.weight")
        print("=============================================")

    return optimizer, model

def training_loop(model, model_args, optimizer, scaler, get_batch_func, C, ctx, device_type, iter_num, best_val_loss, trial=None):
    # Initialize the timer and local iteration number   
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    X, Y = get_batch_func("train")

    while True:
        # Suppress warnings from torch._inductor.utils
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num, C) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate and save checkpoints
        if iter_num % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss(C, model, ctx, get_batch_func, device_type)
                print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if C.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                    })
                # Report intermediate validation loss to Optuna
                if trial is not None and losses["val"] is not None:
                    trial.report(losses["val"], iter_num)
                    print(f"trial.report({losses['val']}, {iter_num})")
                    # Check if the trial should be pruned
                    if trial.should_prune():
                        print("Trial pruned.")
                        raise optuna.TrialPruned()
                # Save the checkpoint if the validation loss is the best so far
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    # Save the checkpoint
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
                    checkpoint_path = os.path.join(C.ckpt_out_dir, "ckpt.pt")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    print(f"Saving checkpoint to {checkpoint_path}...")
                    torch.save(checkpoint, checkpoint_path)
            elif C.always_save_checkpoint:
                # Save checkpoint without validation
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": dict(C),
                }
                checkpoint_path = os.path.join(C.ckpt_out_dir, "ckpt.pt")
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                print(f"Saving checkpoint to {checkpoint_path}...\n")
                torch.save(checkpoint, checkpoint_path)

        if iter_num == 0 and C.eval_only:
            break

        # Training step
        X, Y = get_batch_func("train")
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # Immediately fetch next batch
            X, Y = get_batch_func("train")
            # Backward pass with gradient scaling if using fp16
            scaler.scale(loss).backward()

        # Gradient clipping
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % C.log_interval == 0:
            lossf = loss.item()
            if local_iter_num >= 5:
                mfu = model.estimate_mfu(C.batch_size * C.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"Iter {iter_num}: train loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if iter_num > C.max_iters:
            break
        
    # print best val loss
    print(f"Best val loss for this training: {best_val_loss}\n")

    return best_val_loss

def objective(trial, C):
    """
    Objective function for the hyperparameter search.
    Returns the validation loss for the given hyperparameters.
    """

    # CodeCarbon tracker
    if C.codecarbon:
        # make sure metrics_dir/wandb_project exists
        output_directory = f'{C.metrics_dir}/{C.wandb_project}'
        os.makedirs(output_directory, exist_ok=True)

        tracker = OfflineEmissionsTracker(
            output_dir=output_directory,
            project_name=f'{C.tracker_project}',
            country_iso_code="GBR",  # Replace with your country code
            log_level="error",
        )
        tracker.stop()
        tracker.start()

    C = deepcopy(C)

    if C.wandb_log:
        import wandb
        # Set up W&B run for this trial
        wandb.login(key='5f5e743f253c050dda2db65cbf8864cd444f40b9')
        wandb_run = wandb.init(
            project=C.wandb_project,
            name=f"{C.wandb_run_name}_trial_{trial.number}",
            config=dict(C),
            reinit=True  # Ensure a new run is created for each trial
        )

    # Update configuration for this trial
    C.ckpt_out_dir = os.path.join(C.ckpt_out_dir, f"search_trial_{trial.number}")
    C.learning_rate = trial.suggest_float('learning_rate', 0.0000001, 0.01)
    C.min_lr = C.learning_rate / 10
    C.gradient_accumulation_steps = trial.suggest_categorical('gradient_accumulation_steps', [4, 8, 12, 16, 32])
    C.beta1 = trial.suggest_float('beta1', 0.85, 0.95)
    C.beta2 = trial.suggest_float('beta2', 0.90, 0.999)
    C.weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-1)
    C.dropout = trial.suggest_float('dropout', 0.0, 0.2)
    n_value = trial.suggest_categorical('n_value', [8, 16, 32])
    C.n_head = n_value
    C.n_layer = n_value
    C.n_embd = trial.suggest_categorical('n_embd', [512, 1024, 1536, 2048])
    if C.n_embd % C.n_head != 0:
        if C.wandb_log:
            wandb_run.finish()
        raise optuna.TrialPruned()
    if C.finetune_method == 'LoRA':
        C.LoRA_rank = trial.suggest_int('LoRA_rank', 1, 8)
        C.LoRA_alpha = trial.suggest_int('LoRA_alpha', 1, 8)
    C.grad_clip = trial.suggest_float('grad_clip', 0.0, 10.0)
    C.latent_dim = trial.suggest_categorical('latent_dim', [128, 256, 512, 1024])

    # Device and context setup
    device_type = "cuda" if "cuda" in C.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Load data
    if C.adaptation in ['base', 'cifextd']:
        train_data, val_data, cif_start_indices, cif_start_indices_val, cif_start_indices_underrep, tokenizer, vocab_size, unk_token_id = load_data_txt(C)
    elif C.adaptation == 'regression':
        tokenizer, vocab_size, train_loader, val_loader, test_loader, unk_token_id, max_token_length = load_data_table(C)
    else:
        if C.wandb_log:
            wandb_run.finish()
        raise ValueError(f"Unknown adaptation type: {C.adaptation}")

    # Initialize the model
    model, iter_num, best_val_loss, model_args, checkpoint = initialize_model(C, vocab_size, unk_token_id, max_token_length)
    model.to(C.device)
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    # Implement the finetuning method
    model = implement_finetune_method(C, model, checkpoint)

    # Setup the optimizer
    optimizer, model = optimizer_setup(C, model, checkpoint)

    # Prepare data loader iterators
    loader_iter = {}
    if C.adaptation in ['base', 'cifextd']:
        get_batch_func = lambda split: get_batch(
            split, C, device_type, train_data, val_data, cif_start_indices, cif_start_indices_val,
            cif_start_indices_underrep, loader_iter=loader_iter
        )
    elif C.adaptation == 'regression':
        get_batch_func = lambda split: get_batch(
            split, C, device_type, train_loader=train_loader, val_loader=val_loader, loader_iter=loader_iter
        )

    # Training loop with pruning
    # trial should be pruned if torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 410.00 MiB (GPU 0; 39.38 GiB total capacity; 38.28 GiB already allocated; 155.38 MiB free; 38.72 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
    try:
        best_val_loss = training_loop(model, model_args, optimizer, scaler, get_batch_func, C, ctx, device_type, iter_num, best_val_loss, trial)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error; pruning trial.")
            if C.wandb_log:
                wandb_run.finish()
            raise optuna.TrialPruned()
        else:
            raise e

    # Finish W&B run
    if C.wandb_log:
        wandb_run.finish()

    # Stop CodeCarbon tracker
    if C.codecarbon:
        tracker.stop()

    return best_val_loss