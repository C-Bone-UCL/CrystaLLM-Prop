import os
from dataclasses import dataclass

from contextlib import nullcontext
from omegaconf import OmegaConf
import torch
import pickle
from tqdm import tqdm

import gzip
import re
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

from crystallm import (
    parse_config,
    CIFTokenizer_extd as CIFTokenizer,
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
    pkl_file: str = "CIF_BG_proj/BG_cifs_process_steps/BG_large_test.pkl.gz"
    plot_dir: str = "inference_plots/BG_all_excl_0BG/"

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

    # Do not overwrite vocab_size if it exists
    if 'vocab_size' not in checkpoint_model_args or checkpoint_model_args['vocab_size'] is None:
        checkpoint_model_args['vocab_size'] = meta_vocab_size  # Only set if not present

    print("Model configuration:")
    print(OmegaConf.to_yaml(checkpoint_model_args))

    # Now create the model with the checkpoint's vocab_size
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)

    state_dict = checkpoint["model"]

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

    if C.compile:
        model = torch.compile(model)  # requires PyTorch 2.0 (optional)
        print("Model compiled successfully.")

    # Read the .pkl.gz file
    with gzip.open(C.pkl_file, 'rb') as f:
        data = pickle.load(f)

    # data = data[:2]  # limit the number of samples to 100

    true_bandgaps = []
    predicted_bandgaps = []
    invalid_entries = []

    for idx, entry in enumerate(tqdm(data, desc="Evaluating test set predicted vs true BG values")):
        identifier, cif_content = entry

        # Extract the prompt up to 'Bandgap_eV:'
        # start the prompt at data_ and end it at Bandgap_eV:
        match = re.search(r"(data_[^\n]*\n)[\s\S]*?(Bandgap_eV)", cif_content)
        if match:
            start_index, end_index = match.start(), match.end()
            prompt = cif_content[start_index:end_index]
            prompt = re.sub(r"^[ \t]+|[ \t]+$", "", prompt, flags=re.MULTILINE)

        # Encode the prompt
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]

        # Generate prediction
        with torch.no_grad():
            with ctx:
                try:
                    y = model.generate(x, max_new_tokens=10, temperature=C.temperature, top_k=C.top_k)
                    generated = decode(y[0].tolist())
                except Exception as e:
                    invalid_entries.append(identifier) if identifier not in invalid_entries else None
                    continue

        try:
            # Regex pattern to match the bandgap value
            pattern = r"Bandgap_eV(?:\s+\S+)*?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            match = re.search(pattern, generated)
            if match:
                # Ensure the captured group is not empty
                bandgap_str = match.group(1)
                if bandgap_str:
                    predicted_bandgap = float(bandgap_str)
                else:
                    invalid_entries.append(identifier) if identifier not in invalid_entries else None
                    continue
            else:
                invalid_entries.append(identifier) if identifier not in invalid_entries else None
                continue
        except (ValueError, IndexError) as e:
            invalid_entries.append(identifier) if identifier not in invalid_entries else None
            continue

        # Get true bandgap value
        try:
            true_bandgap_value = float(cif_content.split('Bandgap_eV')[1].strip())
        except Exception as e:
            invalid_entries.append(identifier) if identifier not in invalid_entries else None
            continue

        # Append to lists
        true_bandgaps.append(true_bandgap_value)
        predicted_bandgaps.append(predicted_bandgap)

    print('true_bandgaps:', true_bandgaps)
    print('predicted_bandgaps:', predicted_bandgaps)
    print('number of invalid entries or predictions:', len(invalid_entries))
    print('invalid entries:', invalid_entries)

    # save figure in a new directory called from plot_dir and metrics
    if not os.path.exists(C.plot_dir):
        os.makedirs(C.plot_dir)
    
    mse = mean_squared_error(true_bandgaps, predicted_bandgaps)
    mae = mean_absolute_error(true_bandgaps, predicted_bandgaps)
    rmse = mse ** 0.5

    with open(os.path.join(C.plot_dir, f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, MSE: {mse:.4f}.pkl'), 'wb') as f:
        pickle.dump({'true_bandgaps': true_bandgaps, 'predicted_bandgaps': predicted_bandgaps}, f)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Plot the true vs predicted bandgap values
    plt.figure(figsize=(8, 6))
    plt.scatter(true_bandgaps, predicted_bandgaps, alpha=0.5)
    plt.xlabel('True Bandgap (eV)')
    plt.ylabel('Predicted Bandgap (eV)')
    plt.title('True vs Predicted Bandgap')
    plt.plot([min(true_bandgaps), max(true_bandgaps)],
            [min(true_bandgaps), max(true_bandgaps)], 'r--')
    plt.savefig(os.path.join(C.plot_dir, f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, MSE: {mse:.4f}.png'))
    plt.close()

    print(f"Plot saved successfully to {C.plot_dir}MAE: {mae:.4f}, RMSE: {rmse:.4f}, MSE: {mse:.4f}.png")

