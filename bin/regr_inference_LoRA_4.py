import os
import gzip
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
import math
from torch import nn
from torch.nn import functional as F

from bin.train import TrainDefaults
from crystallm._model import GPT_regression, GPTConfig
from crystallm import CIFTokenizer, parse_config

# Configuration
CHECKPOINT_PATH = 'model_ckpts/regression_models/BG_LoRA/search_trial_4/ckpt.pt'
TEST_DATA_PATH = 'CIF_BG_proj/test_dataset.pkl.gz'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_DIR = 'inference_plots/BG_regr_LoRA/'

# Define the Dataset class
class CIFRegressionDataset(Dataset):
    def __init__(self, dataframe, max_length, unk_token_id):
        self.data = dataframe.reset_index(drop=True)
        self.max_length = max_length
        self.unk_token_id = unk_token_id  # ID for the <unk> token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data.loc[idx, 'CIFs_tokenized']

        x = np.full(self.max_length, self.unk_token_id, dtype=np.int64)
        token_ids = [self.unk_token_id if token == '<unk>' else int(token) for token in tokens]
        x[:min(len(token_ids), self.max_length)] = token_ids[:self.max_length]

        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(self.data.loc[idx, 'Bandgap (eV)'], dtype=torch.float32)
        return x, y

class LinearWithLoRA(nn.Module):
    def __init__(self, old_linear: nn.Linear, rank: int, alpha: int):
        super().__init__()
        # Copy the original weights and bias
        self.weight = nn.Parameter(old_linear.weight.data.clone())
        if old_linear.bias is not None:
            self.bias = nn.Parameter(old_linear.bias.data.clone())
        else:
            self.bias = None

        in_dim, out_dim = old_linear.in_features, old_linear.out_features
        std_dev = 1 / math.sqrt(rank)

        # First set of LoRA parameters directly under self.lora
        self.lora = nn.Module()
        self.lora.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.lora.B = nn.Parameter(torch.zeros(rank, out_dim))

        # Second set of LoRA parameters under self.linear.lora to match desired naming
        self.linear = nn.Module()
        self.linear.lora = nn.Module()
        self.linear.lora.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.linear.lora.B = nn.Parameter(torch.zeros(rank, out_dim))

        # Same alpha scaling for both sets
        self.alpha = alpha

    def forward(self, x):
        # Regular linear operation
        out = x @ self.weight.T
        if self.bias is not None:
            out += self.bias

        # LoRA adjustments: sum both sets of LoRA contributions
        lora_out_1 = self.alpha * (x @ self.lora.A @ self.lora.B)
        lora_out_2 = self.alpha * (x @ self.linear.lora.A @ self.linear.lora.B)

        return out + lora_out_1 + lora_out_2

def get_submodule(model: nn.Module, target_name: str) -> nn.Module:
    """
    Recursively find the submodule given a dotted path.
    """
    names = target_name.split('.')
    m = model
    for name in names:
        if name:
            m = getattr(m, name)
    return m

def replace_linear_with_lora(model: nn.Module, rank: int = 16, alpha: int = 16):
    """
    Replace all nn.Linear layers in the model with LinearWithLoRA,
    now capable of holding two sets of LoRA parameters.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            parent_module = get_submodule(model, parent_name)
            old_linear = getattr(parent_module, module_name)
            lora_linear = LinearWithLoRA(old_linear, rank, alpha)
            setattr(parent_module, module_name, lora_linear)

def initialize_model(checkpoint_path):
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Extract model_args and config from the checkpoint
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        raise KeyError("Checkpoint does not contain 'model_args' key.")
    
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        LoRA_rank = saved_config.get('LoRA_rank')
        LoRA_alpha = saved_config.get('LoRA_alpha')
        FT_method = saved_config.get('finetune_method')
    else:
        raise KeyError("Checkpoint does not contain 'config' key.")

    best_val_loss = checkpoint['best_val_loss']
    print(f'Best val loss from checkpoint: {best_val_loss}\n')

    # Initialize the model configuration and model
    config = GPTConfig(**model_args)
    model = GPT_regression(config)

    # Extract the state dict from the checkpoint
    state_dict = checkpoint['model']

    # 1. Remove '_orig_mod.' prefix
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    # print("State dict keys:", new_state_dict.keys())

    # 2. Replace '.linear.linear.' with '.linear.' 
    temp_state = {}
    for k, v in new_state_dict.items():
        new_k = k.replace('.linear.linear.', '.')
        temp_state[new_k] = v
    new_state_dict = temp_state

    # Apply LoRA modifications if necessary before loading state_dict
    if FT_method == "LoRA":
        if LoRA_rank is None or LoRA_alpha is None:
            raise ValueError("LoRA parameters not found in checkpoint configuration.")
        # This prepares the model to have the LoRA parameters in place
        replace_linear_with_lora(model, rank=LoRA_rank, alpha=LoRA_alpha)

    # print expected keys
    # print("Expected keys:", model.state_dict().keys())
    # print new state dict keys
    # print("New state dict keys:", new_state_dict.keys())

    # Now load all parameters, including LoRA parameters, into the model
    checkpoint_load = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", checkpoint_load.missing_keys)
    print("Unexpected keys:", checkpoint_load.unexpected_keys)
    print("Model loaded successfully with all LoRA parameters.")

    # Move the model to the device and set to eval mode
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path} and moved to {DEVICE}.")
    return model

def load_test_data(test_data_path):
    with gzip.open(test_data_path, 'rb') as f:
        test_df = pickle.load(f)
    print(f"Loaded test dataset with {len(test_df)} samples.")
    # use only first 12 samples for testing
    # test_df = test_df[:12]
    return test_df

def perform_inference(model, test_loader):
    all_preds = []
    all_trues = []

    with torch.no_grad():
         for batch in tqdm(test_loader, desc="Performing Inference", unit="batch"):
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds, _ = model(inputs)
            preds = preds.squeeze(-1)  # Shape: (batch_size,)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(targets.cpu().numpy())
    print("Inference completed on test dataset.")
    # Round values for printing
    all_trues = [round(val, 3) for val in all_trues]
    all_preds = [round(val, 3) for val in all_preds]
    print(f"True values: {all_trues[:5]} ...")
    print(f"Predicted values: {all_preds[:5]} ...")
    # Compute metrics
    mse = np.mean((np.array(all_trues) - np.array(all_preds)) ** 2)
    mae = np.mean(np.abs(np.array(all_trues) - np.array(all_preds)))
    print(f"MSE: {mse:.3f}, MAE: {mae:.3f}")

    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    with open(os.path.join(PLOT_DIR, f'MAE_{mae:.4f}_MSE_{mse:.4f}.pkl'), 'wb') as f:
        pickle.dump({'true_bandgaps': all_trues, 'predicted_bandgaps': all_preds}, f)

    return all_trues, all_preds, mae, mse

def plot_true_vs_predicted(true_values, predicted_values, mae, mse):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=true_values, y=predicted_values, alpha=0.6, edgecolor=None)
    max_val = max(max(true_values), max(predicted_values))
    min_val = min(min(true_values), min(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('True Bandgap (eV)', fontsize=14)
    plt.ylabel('Predicted Bandgap (eV)', fontsize=14)
    plt.title('True vs. Predicted Bandgap Values', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'MAE_{mae:.4f}_MSE_{mse:.4f}.png'))
    plt.close()

def main():
    # Load the test dataset
    test_df = load_test_data(TEST_DATA_PATH)

    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Extract model_args and configuration from the checkpoint
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        raise KeyError("Checkpoint does not contain 'model_args' key.")

    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        FT_method = saved_config.get('finetune_method')
    else:
        raise KeyError("Checkpoint does not contain 'config' key.")

    print(f"Model configuration: {model_args}")

    # Instantiate the tokenizer and get the <unk> token ID
    tokenizer = CIFTokenizer()
    unk_token_id = tokenizer.token_to_id["<unk>"]

    # Determine max_token_length
    max_token_length = model_args.get('max_token_length', 6529)

    # Initialize the Dataset and DataLoader
    test_dataset = CIFRegressionDataset(test_df, max_length=max_token_length, unk_token_id=unk_token_id)
    batch_size = saved_config.get('batch_size', 1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test DataLoader created with batch size {batch_size}.")

    # Initialize the model and load trained weights
    model = initialize_model(CHECKPOINT_PATH)

    # Perform inference
    true_vals, pred_vals, mae, mse = perform_inference(model, test_loader)

    # Plot the results
    plot_true_vs_predicted(true_vals, pred_vals, mae, mse)

def run_inference():
    main()

if __name__ == "__main__":
    run_inference()
