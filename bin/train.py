"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
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
from dataclasses import dataclass
from contextlib import nullcontext

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import logging
import numpy as np
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf

from codecarbon import OfflineEmissionsTracker
from carbontracker.tracker import CarbonTracker
from carbontracker import parser

from crystallm import (
    parse_config,
    CIFTokenizer,
    GPT,
    GPTConfig,
    GPT_regression,
    _model,
)

importlib.reload(_model)

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
    adaptation: str = "base" # Options: 'base', 'regression'

    # Sanity check
    sanity_check: bool = False # if True, print parameter changes after update

    # Train-test split for regression adaptation
    test_size: float = 0.05
    val_size: float = 0.05

def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> torch.Tensor | None:
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

if __name__ == "__main__":
    # Load configuration
    C = parse_config(TrainDefaults)
    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    # CodeCarbon tracker
    if C.codecarbon:
        # make sure metrics_dir/wandb_project exists
        output_directory=f'{C.metrics_dir}/{C.wandb_project}'
        os.makedirs(output_directory, exist_ok=True)

        tracker = OfflineEmissionsTracker(
        output_dir=output_directory,
        project_name=f'{C.tracker_project}',
        country_iso_code="GBR",  # Replace with your country code
        )
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

    if C.adaptation == "base":
        train_data = np.memmap(os.path.join(C.dataset, "train.bin"), dtype=np.uint16, mode="r")
        val_data = np.memmap(os.path.join(C.dataset, "val.bin"), dtype=np.uint16, mode="r") if C.validate else None

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

    elif C.adaptation == "regression":
        # Load the DataFrame
        with gzip.open(C.dataset, 'rb') as f:
            df = pickle.load(f)

        train_df, test_df = train_test_split(df, test_size=C.test_size, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=C.val_size / (1 - C.test_size), random_state=42)

        print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Test samples: {len(test_df)}")

        # Create Datasets
        class CIFRegressionDataset(Dataset):
            def __init__(self, dataframe, block_size):
                self.data = dataframe.reset_index(drop=True)
                self.block_size = block_size

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                tokens = self.data.loc[idx, 'CIFs_tokenized']
                x = tokens[:self.block_size]
                # Pad or truncate x to block_size
                x = x + [0] * (self.block_size - len(x)) if len(x) < self.block_size else x[:self.block_size]
                x = torch.tensor(x, dtype=torch.long)
                y = torch.tensor(self.data.loc[idx, 'Bandgap (eV)'], dtype=torch.float32)
                return x, y

        train_dataset = CIFRegressionDataset(train_df, C.block_size)
        val_dataset = CIFRegressionDataset(val_df, C.block_size)
        test_dataset = CIFRegressionDataset(test_df, C.block_size)

        # save the test dataset by going to one level above the dataset directory and saving it
        test_out_dir = os.path.dirname(C.dataset)
        test_out_path = os.path.join(test_out_dir, "test_dataset.pkl.gz")
        # Save the test dataset gzipping it
        with gzip.open(test_out_path, 'wb') as f:
            pickle.dump(test_df, f)
        print(f"Saved test dataset to {test_out_path}")

        # Determine vocab_size from the CIFs_tokenized column
        all_tokens = set(token for tokens in df['CIFs_tokenized'] for token in tokens)
        vocab_size = max(len(all_tokens), 371)
        print(f"Found {len(all_tokens)} unique tokens in the dataset (using vocab_size = {vocab_size})")

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=C.batch_size, shuffle=False)


    def get_batch(split):
        if C.adaptation == 'base':
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
            try:
                x, y = next(loader_iter[split])
            except (KeyError, StopIteration):
                loader_iter[split] = iter(loader)
                x, y = next(loader_iter[split])

        if device_type == "cuda":
            x = x.pin_memory().to(C.device, non_blocking=True)
            y = y.pin_memory().to(C.device, non_blocking=True)
        else:
            x = x.to(C.device)
            y = y.to(C.device)

        return x, y

    # Initialize iterators for loaders
    loader_iter = {}

    # Determine vocab_size
    if C.adaptation == 'base':
        meta_path = os.path.join(C.dataset, "meta.pkl")
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")
    elif C.adaptation == 'regression':
        meta_vocab_size = vocab_size
        print(f"Using vocab_size = {meta_vocab_size} for regression adaptation")

    # Initialize model arguments
    model_args = dict(
        n_layer=C.n_layer,
        n_head=C.n_head,
        n_embd=C.n_embd,
        block_size=C.block_size,
        bias=C.bias,
        vocab_size=meta_vocab_size or 371,
        dropout=C.dropout,
    )

    if C.init_from == "scratch":
        print("Initializing a new model from scratch with a vocab size of 371...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf) if C.adaptation == 'base' else GPT_regression(gptconf) if C.adaptation == 'regression' else None

    elif C.init_from == "resume":
        print(f"Resuming training from {C.out_dir}...")
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]

        # Update model arguments based on checkpoint
        model_args.update(checkpoint_model_args)
        model_args["finetune_method"] = C.finetune_method
        model_args["sanity_check"] = C.sanity_check
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf) if C.adaptation == 'base' else GPT_regression(gptconf) if C.adaptation == 'regression' else None

        # Load state_dict and adjust as needed then remove unwanted prefixes
        state_dict = checkpoint["model"]
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        if C.adaptation == 'regression':
            # Remove outdated 'lm_head' keys
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("lm_head")}
            print("Removed outdated 'lm_head' keys from state_dict.")

        # Load the state_dict with strict=False to ignore missing keys
        model.load_state_dict(state_dict, strict=False)

        # Check for vocabulary size mismatch and resize if necessary
        checkpoint_vocab_size = checkpoint_model_args.get("vocab_size")
        if checkpoint_vocab_size <= meta_vocab_size:
            print(f"Vocabulary size mismatch: checkpoint has {checkpoint_vocab_size}, dataset has {meta_vocab_size}")
            model.resize_token_embeddings(meta_vocab_size)
            model.transformer.wte.weight.grad = None  # Clear gradients for resized embeddings
        elif checkpoint_vocab_size > meta_vocab_size:
            print(f"Vocabulary size mismatch: checkpoint has {checkpoint_vocab_size}, dataset has {meta_vocab_size}")
            raise ValueError("the new dataset has a smaller vocab size than the checkpoint which causes tensor errors")

        iter_num = 0
        best_val_loss = checkpoint.get("best_val_loss", 1e9)

        if C.finetune_method == 'LoRA':
            print("Replacing linear layers with LoRA layers")
            model.replace_linear_with_lora(rank=C.LoRA_rank, alpha=C.LoRA_alpha)
    else:
        raise ValueError(f"Unknown init_from value: {C.init_from}")


    # crop down the model block size to reduce context if desired, using model surgery
    if C.block_size < model.config.block_size:
        model.crop_block_size(C.block_size)
        model_args["block_size"] = C.block_size  # so that the checkpoint will have the right value
    # move the model to the correct device
    model.to(C.device)
    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    # Finetuning methods
    if C.init_from == "resume" and C.finetune_method == 'finetune_head':
        print("Finetuning head only: Freezing transformer layers")
        for param in model.transformer.parameters():
            param.requires_grad = False
        model.transformer.wte.weight.requires_grad = True

    if C.init_from == "resume" and C.finetune_method == 'finetune_all':
        print("Finetuning all layers")
        for param in model.parameters():
            param.requires_grad = True

    if C.init_from == "resume" and C.finetune_method == 'LoRA':
        print("Setting requires_grad for LoRA parameters")
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

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
                print(f"Parameter {param_name} missing in checkpoint.")
                return False  # Parameter was added
            if param_tensor.shape != checkpoint_state[param_name].shape:
                print(f"Parameter {param_name} shape mismatch: {param_tensor.shape} vs {checkpoint_state[param_name].shape}")
                return False  # Parameter shape has changed

        # Check for any extra parameters in the checkpoint that aren't in the model
        for param_name in checkpoint_state:
            if param_name not in model_state:
                print(f"Checkpoint parameter {param_name} missing in current model.")
                return False  # Parameter was removed

        # If all parameters and shapes match
        return True

    # Optimizer setup - we use Adam with weight decay
    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))

    # Load optimizer state from checkpoint if model structure is unmodified, otherwise reset optimizer
    if C.init_from == "resume":
        model_structure_unmodified = is_model_structure_unmodified(model, checkpoint)
        if model_structure_unmodified:
            print("Loading optimizer state from checkpoint...")
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("Model structure modified; resetting optimizer.")
    else:
        print("Initializing optimizer from scratch.")

    # Compile the model if desired
    if C.compile:
        print("Compiling the model (takes a ~minute)...")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # Print a summary of trainable versus frozen parameters
    if C.init_from == "resume" and C.finetune_method == 'LoRA':
        print("Setting requires_grad for LoRA parameters")
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

    # Print a summary of trainable versus frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({100.0 * trainable_params / total_params:.2f}%)")

    if C.sanity_check:
        # Sanity Check: Ensure that the correct parameters are set as trainable - CB: added this
        print("===== Sanity Check: Trainable Parameters =====")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Trainable: {name}")
            else:
                print(f"Frozen: {name}")
        # check in for name, param in model.lm_head.named_parameters(): if 'lm_head' in name:
        if C.adaptation == 'regression':
            for name, param in model.lm_head.named_parameters():
                if 'lm_head' in name:
                    if param.requires_grad:
                        print(f"Trainable: {name}")
                    else:
                        print(f"Frozen: {name}")
        elif C.adaptation == 'base' and C.finetune_method != 'LoRA':
            # check if self.lm_head.weight.requires_grad:
            if model.lm_head.weight.requires_grad:
                print("Trainable: _orig_mod.lm_head.weight")
            else:
                print("Frozen: _orig_mod.lm_head.weight")
        print("=============================================")
            

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, eval_iters in [("train", C.eval_iters_train), ("val", C.eval_iters_val)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
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

    # wandb logging
    if C.wandb_log:
        import wandb
        wandb.login(key='5f5e743f253c050dda2db65cbf8864cd444f40b9')
        wandb.init(project=C.wandb_project, name=C.wandb_run_name + str(time.time()), config=dict(C))
        config = OmegaConf.to_container(C)

    # Training loop
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    if C.sanity_check:
        # Save parameters before the update (for sanity check)
        param_updates = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}

    if C.validate:
        # Load the checkpoint from the output directory
        os.makedirs(C.ckpt_out_dir, exist_ok=True)
        ckpt_out_path = os.path.join(C.ckpt_out_dir, "ckpt.pt")
        if os.path.exists(ckpt_out_path):
            checkpoint_out = torch.load(ckpt_out_path, map_location=C.device)
            best_val_loss_out = checkpoint_out.get("best_val_loss", 1e9)
            print(f'Best val loss from checkpoint: {best_val_loss_out}')
        else:
            best_val_loss_out = 1e9
            print(f'No checkpoint found at {ckpt_out_path}, so best_val_loss_out is set to {best_val_loss_out}')

    print("Warning: torch._inductor.utils warnings are suppressed")
    while True:
        # Suppress warnings from torch._inductor.utils
        logging.getLogger("torch._inductor.utils").setLevel(logging.ERROR)

        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate and save checkpoints
        if iter_num % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss()
                print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if C.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                    })
                if losses["val"] < best_val_loss_out or losses["val"] < best_val_loss:
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
                print(f"Saving checkpoint to {checkpoint_path}...")
                torch.save(checkpoint, checkpoint_path)

        if iter_num == 0 and C.eval_only:
            break

        # Training step
        X, Y = get_batch("train")
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # Immediately fetch next batch
            X, Y = get_batch("train")
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

    # Finalize tracker
    if C.codecarbon:
        tracker.stop()



