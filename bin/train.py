"""
Adapted from:
https://github.com/karpathy/nanoGPT/blob/eba36e84649f3c6d840a93092cb779a260544d08/train.py
"""
import os
from dataclasses import dataclass
from typing import Union
import math
import time

from crystallm import parse_config
from omegaconf import OmegaConf
import numpy as np
import torch
import pickle
from contextlib import nullcontext

from crystallm import (
    GPT,
    GPTConfig,
)

@dataclass
class TrainDefaults:
    out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored
    ckpt_out_dir: str = "out"  # the path to the folder where the model checkpoints will be stored (CB: added this)
    eval_interval: int = 250  # how often to evaluate against the validation set
    log_interval: int = 1  # how often to print to
    eval_iters_train: int = 200
    eval_iters_val: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = False  # if True, always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume'

    # wandb logging
    wandb_log: bool = False # disabled by default
    wandb_project:str = 'crystallm_CIF_BG'
    wandb_run_name:str = 'BG_large'

    # data
    dataset: str = ""  # the path to the folder containing the .bin files with encoded tokens
    gradient_accumulation_steps: int = 40  # used to simulate larger batch sizes
    batch_size: int = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size: int = 2048  # context of up to `block_size` previous characters

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = False  # do we use bias inside LayerNorm and Linear layers?

    # AdamW optimizer
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95  # make a bit bigger because number of tokens per iter is small
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for; not super necessary potentially
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster
    underrep_p: float = 0.0
    validate: bool = False  # whether to evaluate the model using the validation set

    # finetune argument (CB: added this)
    finetune_method: str = 'finetune_all'  # finetune the model on the dataset (CB: added this) (ex: 'freeze_head', 'finetune_all', 'LoRA')

def read_start_indices(
    max_start_index: int,
    data_dir: str,
    starts_fname: str,
    on_condition: bool = True,
    required: bool = False,
) -> Union[torch.Tensor, None]:
    start_indices = None
    starts_path = os.path.join(data_dir, starts_fname)
    if on_condition:
        if os.path.exists(starts_path):
            print(f"Reading start indices from {starts_path}...")
            with open(starts_path, "rb") as f:
                start_indices = torch.tensor(pickle.load(f))  # should be sorted
            # remove indices that would result in out-of-bounds sequences
            start_indices = start_indices[start_indices <= max_start_index]
        elif required:
            raise Exception(f"Expected to find a file in dataset dir named '{starts_fname}'")
    return start_indices


if __name__ == "__main__":
    C = parse_config(TrainDefaults)

    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    print(f"Creating {C.out_dir}...")
    os.makedirs(C.out_dir, exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if not C.dataset:
        raise Exception("The 'dataset' option is required and cannot be empty")

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

    def get_batch(split):
        data = train_data if split == "train" else val_data

        ix = torch.randint(len(data) - C.block_size, (C.batch_size,))
        if split == "train":
            if C.underrep_p is not None and np.random.random() < C.underrep_p:
                ix = cif_start_indices_underrep[torch.randperm(len(cif_start_indices_underrep))[:C.batch_size]]
            elif cif_start_indices is not None:
                ix = cif_start_indices[torch.randperm(len(cif_start_indices))[:C.batch_size]]
        elif cif_start_indices_val is not None:
            ix = cif_start_indices_val[torch.randperm(len(cif_start_indices_val))[:C.batch_size]]

        x = torch.stack([torch.from_numpy((data[i:i + C.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + C.block_size]).astype(np.int64)) for i in ix])

        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(C.device, non_blocking=True), y.pin_memory().to(C.device, non_blocking=True)
        else:
            x, y = x.to(C.device), y.to(C.device)
        return x, y

    iter_num = 0
    best_val_loss = 1e9

    meta_path = os.path.join(C.dataset, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"Found vocab_size = {meta_vocab_size} (inside {meta_path})")

    model_args = dict(n_layer=C.n_layer, n_head=C.n_head, n_embd=C.n_embd, block_size=C.block_size,
                      bias=C.bias, vocab_size=None, dropout=C.dropout)
    if C.init_from == "scratch":
        print("Initializing a new model from scratch...")
        if meta_vocab_size is None:
            print("Defaulting to vocab_size of 371...")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 371
        gptconf = GPTConfig(**model_args)

    if C.init_from == "resume":
        # Load checkpoint
        print(f"Resuming training from {C.out_dir}...")
        ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=C.device)
        checkpoint_model_args = checkpoint["model_args"]
        
        # Update model arguments based on checkpoint
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        
        model_args["finetune_method"] = C.finetune_method
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)

        # Load state_dict and adjust as needed
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        model.load_state_dict(state_dict)

        # Check for vocabulary size mismatch and resize if necessary
        checkpoint_vocab_size = checkpoint_model_args.get("vocab_size", None)
        if checkpoint_vocab_size is not None and meta_vocab_size is not None:
            if checkpoint_vocab_size != meta_vocab_size:
                print(f"Vocabulary size mismatch detected: checkpoint has {checkpoint_vocab_size}, dataset has {meta_vocab_size}")
                print(f"Resizing token embeddings from {checkpoint_vocab_size} to {meta_vocab_size}")
                model.resize_token_embeddings(meta_vocab_size)
                model.transformer.wte.weight.grad = None  # Clear gradients for resized embeddings

        iter_num = 0
        best_val_loss = checkpoint["best_val_loss"]


    # crop down the model block size if desired, using model surgery
    if C.block_size < model.config.block_size:
        model.crop_block_size(C.block_size)
        model_args["block_size"] = C.block_size  # so that the checkpoint will have the right value
    model.to(C.device)

    # initialize a GradScaler; if enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(C.dtype == "float16"))

    # Finetuning methods (CB: added this)
    if C.init_from == "resume" and C.finetune_method == 'freeze_head':
        print("Finetuning head only: Freezing transformer layers")
        for param in model.transformer.parameters():
            param.requires_grad = False
        # Ensure both tied weights are set correctly
        model.transformer.wte.weight.requires_grad = True
        model.lm_head.weight.requires_grad = True

    if C.init_from == "resume" and C.finetune_method == 'finetune_all':
        print("Finetuning all layers")
        for param in model.parameters():
            param.requires_grad = True

    if C.init_from == "resume" and C.finetune_method == 'LoRA':
        print("Finetuning using LoRA")
        # With LoRA, only LoRA-marked parameters are trainable
        for name, param in model.named_parameters():
            param.requires_grad = 'lora' in name

    # optimizer
    optimizer = model.configure_optimizers(C.weight_decay, C.learning_rate, (C.beta1, C.beta2))
    if C.init_from == "resume":
        print("Loading optimizer state from checkpoint... to avoid tensor mismatch errors")
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Remove optimizer state entries for parameters whose sizes have changed
        def _optimizer_state_shape_mismatch(param, opt_state_param):
            for state_key, state_value in opt_state_param.items():
                if torch.is_tensor(state_value):
                    if state_value.shape != param.shape:
                        return True
            return False

        state = optimizer.state
        param_keys = list(state.keys())
        for param in param_keys:
            if _optimizer_state_shape_mismatch(param, state[param]):
                print(f"Removing optimizer state for parameter with shape mismatch: {param.shape}")
                del state[param]

    if C.compile:
        print("Compiling the model (takes a ~minute)...")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # # Sanity Check: Ensure that the correct parameters are set as trainable - CB: added this
    print("===== Sanity Check: Trainable Parameters =====")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable: {name}")
        else:
            print(f"Frozen: {name}")

    # Print a summary of trainable versus frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params} ({100.0 * trainable_params / total_params:.2f}%)")

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

    # training loop
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    running_mfu = -1.0

    # Dictionary to store parameter states before update (CB: added this)
    # Save parameters before the update (for sanity check)
    if iter_num == 0:
        param_updates = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Remove '_orig_mod.' prefix if it exists when saving
                name_without_prefix = name.replace('_orig_mod.', '')
                param_updates[name_without_prefix] = param.clone().detach()

    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if C.decay_lr else C.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % C.eval_interval == 0:
            if C.validate:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if C.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train_loss": losses["train"],
                        "val_loss": losses["val"],
                        "lr": lr,
                    })
            if (C.validate and losses["val"] < best_val_loss) or C.always_save_checkpoint:
                best_val_loss = losses["val"] if C.validate else 0.
                if iter_num > 0:
                    model_args['vocab_size'] = model.config.new_vocab_size
                    checkpoint = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": dict(C),
                    }
                    # save the checkpoint (CB: added this to handle ckpt out paths)
                    checkpoint_path = os.path.join(C.ckpt_out_dir, "ckpt.pt") if C.ckpt_out_dir else os.path.join(C.out_dir, "ckpt.pt")
                    #make sure the directory exists
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    print(f"saving checkpoint to {checkpoint_path}...")
                    torch.save(checkpoint, checkpoint_path)
        if iter_num == 0 and C.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(C.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if C.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()

        # After optimizer step, print parameter differences (sanity check)
        if iter_num == 0:
            print("===== Sanity Check: Parameter Changes After Update =====")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Remove '_orig_mod.' prefix to match saved names
                    name_without_prefix = name.replace('_orig_mod.', '')

                    if name_without_prefix in param_updates:
                        old_param = param_updates[name_without_prefix]
                        update = param - old_param
                        update_norm = update.norm().item()
                        print(f"Parameter: {name}, Update Norm: {update_norm}")
                    else:
                        print(f"Warning: Parameter '{name}' not found in saved param_updates. Skipping.")
            print("=========================================================")

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % C.log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = model.estimate_mfu(C.batch_size * C.gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > C.max_iters:
            break
