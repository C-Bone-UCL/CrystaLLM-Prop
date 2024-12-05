import os
from dataclasses import dataclass

from contextlib import nullcontext
from omegaconf import OmegaConf
import torch
import pickle

from crystallm import (
    parse_config,
    CIFTokenizer,
    GPT,
    GPTConfig,
)


@dataclass
class SampleDefaults:
    out_dir: str = "out"  # the path to the directory containing the trained model
    start: str = "\n"  # the prompt; can also specify a file, use as: "FILE:prompt.txt"
    num_samples: int = 2  # number of samples to draw
    max_new_tokens: int = 3000  # number of tokens generated in each sample
    temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k: int = 10  # retain only the top_k most likely tokens, clamp others to have 0 probability
    seed: int = 1337
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    dtype: str = "bfloat16"  # 'float32' or 'bfloat16' or 'float16'
    compile: bool = False  # use PyTorch 2.0 to compile the model to be faster
    target: str = "console"  # where the generated content will be sent; can also be 'file'
    generated_dir: str = "generated_cifs"
    token_resize: bool = False  # resize token embeddings to match the checkpoint's vocab size - if set top true, need to provide dataset path
    dataset: str = "data"  # the path to the directory containing the dataset

def load_state_dict_LoRA(model, state_dict):
    model_state_dict = model.state_dict()

    # Filter out keys that are in the model state_dict but not in the provided state_dict
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Load the filtered state dict with strict=False to ignore missing LoRA keys
    model.load_state_dict(filtered_state_dict, strict=False)

    # Initialize any remaining LoRA parameters that were not in the state_dict
    for name, param in model.named_parameters():
        if name not in filtered_state_dict:
            param.data.uniform_(-0.01, 0.01)  # You may adjust this initialization as needed.

    print("Loaded state_dict with LoRA compatibility.")

if __name__ == "__main__":
    C = parse_config(SampleDefaults)

    print("Using configuration:")
    print(OmegaConf.to_yaml(C))

    torch.manual_seed(C.seed)
    torch.cuda.manual_seed(C.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in C.device else "cpu"  # for later use in torch.autocast
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    #load meta vocab size
    if C.token_resize and C.dataset:
        meta_path = os.path.join(C.dataset, "meta.pkl")
        meta_vocab_size = None
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            meta_vocab_size = meta["vocab_size"]
            print(f"Found dataset vocab_size = {meta_vocab_size} (inside {meta_path})")

    # Load the checkpoint
    ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=C.device)
    checkpoint_model_args = checkpoint["model_args"]
    checkpoint_model_args['vocab_size'] = meta_vocab_size  # Ensure vocab_size is set
    print("Model configuration:")
    print(OmegaConf.to_yaml(checkpoint_model_args))
    state_dict = checkpoint["model"]

    # Now create the model with the correct vocab_size
    gptconf = GPTConfig(**checkpoint_model_args)
    # Create model with the config from the checkpoint
    model = GPT(gptconf)

    # Fix any unwanted prefix in state_dict
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    # Load the state dict into the model (CB: added this) for LoRA compatibility
    try:
        if model.config.finetune_method == "LoRA":
            load_state_dict_LoRA(model, state_dict)
        else:
            model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")

    # Ensure the model is on the correct device and in evaluation mode
    model.eval()
    model.to(C.device)

    # print the best model configuration loss
    # Print the best validation loss from the checkpoint
    best_val_loss = checkpoint.get("best_val_loss", None)
    if best_val_loss is not None:
        print(f"Best validation loss (from checkpoint): {best_val_loss:.4f}")
    else:
        print("Validation loss not found in checkpoint.")

    # Optionally print the train loss if it was saved (update this if you stored it in the checkpoint)
    train_loss = checkpoint.get("train_loss", None)
    if train_loss is not None:
        print(f"Training loss (from checkpoint): {train_loss:.4f}")
    else:
        print("Training loss not found in checkpoint.")

    if C.compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)
        print("Model compiled successfully.")

    # Encode the prompt
    prompt = C.start
    if prompt.startswith("FILE:"):
        with open(prompt[5:], "r", encoding="utf-8") as f:
            prompt = f.read()
    start_ids = encode(tokenizer.tokenize_cif(prompt))
    x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]

    # Run generation
    with torch.no_grad():
        with ctx:
            print("Generating samples...")
            for k in range(C.num_samples):
                try:
                    y = model.generate(x, C.max_new_tokens, temperature=C.temperature, top_k=C.top_k)
                    generated = decode(y[0].tolist())
                except Exception as e:
                    print(f"Error generating sample: {e}")
                    continue

                if C.target == "console":
                    print(generated)
                    print('---------------')
                elif C.target == "file":
                    os.makedirs(C.generated_dir, exist_ok=True)
                    fname = os.path.join(C.generated_dir, f"sample_{k+1}.cif")
                    print(f"writing generated content to {fname} ...")
                    with open(fname, "wt") as f:
                        f.write(generated)
