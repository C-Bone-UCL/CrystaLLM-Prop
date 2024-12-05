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

# Configuration
@dataclass
class SampleDefaults:
    out_dir: str = "out"  # Path to the directory containing the trained model
    device: str = "cuda"  # 'cpu' or 'cuda'
    dtype: str = "bfloat16"  # 'float32', 'bfloat16', or 'float16'
    compile: bool = False  # Use PyTorch 2.0 to compile the model
    pkl_file: str = "CIF_BG_proj/BG_cifs_process_steps/BG_large_test.pkl.gz"
    plot_dir: str = "inference_plots/BG_all_excl_0BG/"

def load_test_data(pkl_file):
    with gzip.open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def initialize_model(ckpt_path, device):
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    config = checkpoint.get("config", {})
    finetune_method = config.get('finetune_method')
    LoRA_rank = config.get('LoRA_rank')
    LoRA_alpha = config.get('LoRA_alpha')

    # Initialize the model configuration
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)

    # Apply LoRA modifications if necessary
    if finetune_method == "LoRA":
        if LoRA_rank is None or LoRA_alpha is None:
            raise ValueError("LoRA parameters not found in checkpoint configuration.")
        model.replace_linear_with_lora(rank=LoRA_rank, alpha=LoRA_alpha)

    # Load the state dict
    state_dict = checkpoint["model"]
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # Adjust the state_dict keys to match the model's parameter names
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'linear.linear' in k:
            new_key = k.replace('linear.linear', 'linear')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    # Load the state dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    # Move the model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model

def main():
    C = parse_config(SampleDefaults)
    print("Using configuration:")
    print(C)

    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in C.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[C.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    tokenizer = CIFTokenizer()
    encode = tokenizer.encode
    decode = tokenizer.decode

    # Initialize the model and load trained weights
    ckpt_path = os.path.join(C.out_dir, "ckpt.pt")
    model = initialize_model(ckpt_path, C.device)

    # Load the test data
    data = load_test_data(C.pkl_file)

    true_bandgaps = []
    predicted_bandgaps = []
    invalid_entries = []

    for idx, entry in enumerate(tqdm(data, desc="Evaluating test set predicted vs true BG values")):
        identifier, cif_content = entry

        # Extract the prompt up to 'Bandgap_eV:'
        match = re.search(r"(data_[^\n]*\n)[\s\S]*?(Bandgap_eV)", cif_content)
        if match:
            start_index, end_index = match.start(), match.end()
            prompt = cif_content[start_index:end_index]
            prompt = re.sub(r"^[ \t]+|[ \t]+$", "", prompt, flags=re.MULTILINE)
        else:
            invalid_entries.append(identifier)
            continue

        # Encode the prompt
        start_ids = encode(tokenizer.tokenize_cif(prompt))
        x = torch.tensor(start_ids, dtype=torch.long, device=C.device)[None, ...]

        # Generate prediction
        with torch.no_grad():
            with ctx:
                try:
                    y = model.generate(x, max_new_tokens=4, temperature=0.7, top_k=10)
                    generated = decode(y[0].tolist())
                except Exception as e:
                    invalid_entries.append(identifier)
                    continue

        try:
            # Regex pattern to match the bandgap value
            pattern = r"Bandgap_eV(?:\s+\S+)*?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            match = re.search(pattern, generated)
            if match:
                bandgap_str = match.group(1)
                if bandgap_str:
                    predicted_bandgap = float(bandgap_str)
                else:
                    invalid_entries.append(identifier)
                    continue
            else:
                invalid_entries.append(identifier)
                continue
        except (ValueError, IndexError) as e:
            invalid_entries.append(identifier)
            continue

        # Get true bandgap value
        try:
            true_bandgap_value = float(cif_content.split('Bandgap_eV')[1].strip())
        except Exception as e:
            invalid_entries.append(identifier)
            continue

        # Append to lists
        true_bandgaps.append(true_bandgap_value)
        predicted_bandgaps.append(predicted_bandgap)

    print('true_bandgaps:', true_bandgaps)
    print('predicted_bandgaps:', predicted_bandgaps)
    print('number of invalid entries or predictions:', len(invalid_entries))
    print('invalid entries:', invalid_entries)

    # Save figure and metrics
    if not os.path.exists(C.plot_dir):
        os.makedirs(C.plot_dir)

    mse = mean_squared_error(true_bandgaps, predicted_bandgaps)
    mae = mean_absolute_error(true_bandgaps, predicted_bandgaps)
    rmse = mse ** 0.5

    with open(os.path.join(C.plot_dir, f'MAE_{mae:.4f}_RMSE_{rmse:.4f}_MSE_{mse:.4f}.pkl'), 'wb') as f:
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
    plt.savefig(os.path.join(C.plot_dir, f'MAE_{mae:.4f}_RMSE_{rmse:.4f}_MSE_{mse:.4f}.png'))
    plt.close()

    print(f"Plot saved successfully to {os.path.join(C.plot_dir, f'MAE_{mae:.4f}_RMSE_{rmse:.4f}_MSE_{mse:.4f}.png')}")

if __name__ == "__main__":
    main()
