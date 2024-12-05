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

from bin.train import TrainDefaults
from crystallm._model import GPT_regression, GPTConfig
from crystallm import CIFTokenizer, parse_config

# Configuration
CHECKPOINT_PATH = 'model_ckpts/regression_models/BG_LoRA_test/search_trial_1/ckpt.pt'  #
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

def load_test_data(test_data_path):
    with gzip.open(test_data_path, 'rb') as f:
        test_df = pickle.load(f)
    print(f"Loaded test dataset with {len(test_df)} samples.")
    return test_df

def initialize_model(checkpoint_path):
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Extract model_args from the checkpoint
    if 'model_args' in checkpoint:
        model_args = checkpoint['model_args']
    else:
        raise KeyError("Checkpoint does not contain 'model_args' key. Ensure the checkpoint format is correct.")
    
    # Extract the configuration from the checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Extract LoRA parameters and finetuning method
        LoRA_rank = saved_config.get('LoRA_rank')
        LoRA_alpha = saved_config.get('LoRA_alpha')
        FT_method = saved_config.get('finetune_method')
    else:
        raise KeyError("Checkpoint does not contain 'config' key. Ensure the checkpoint format is correct.")

    # Initialize the model configuration
    config = GPTConfig(**model_args)
    # Initialize the model
    model = GPT_regression(config)
    
    # Apply LoRA modifications if necessary
    if FT_method == "LoRA":
        if LoRA_rank is None or LoRA_alpha is None:
            raise ValueError("LoRA parameters not found in checkpoint configuration.")
        model.replace_linear_with_lora(rank=LoRA_rank, alpha=LoRA_alpha)

    # Load the state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # Remove '_orig_mod.' from the keys of the state_dict
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        raise KeyError("Checkpoint does not contain 'model' key. Ensure the checkpoint format is correct.")
    
    # Adjust the state_dict keys to match the model's parameter names
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'linear.linear' in k:
            new_key = k.replace('linear.linear', 'linear')
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    print(state_dict.keys())
    try:
        # Load the state dict into the model
        model.load_state_dict(new_state_dict, strict=False)
        print("Model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")

    # Move the model to the specified device and set to evaluation mode
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path} and moved to {DEVICE}.")
    return model

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

    # Save results
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    with open(os.path.join(PLOT_DIR, f'MAE_{mae:.4f}_MSE_{mse:.4f}.pkl'), 'wb') as f:
        pickle.dump({'true_bandgaps': all_trues, 'predicted_bandgaps': all_preds}, f)

    return all_trues, all_preds, mae, mse

def plot_true_vs_predicted(true_values, predicted_values, mae, mse):
    sns.set_theme(style='whitegrid')
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=true_values, y=predicted_values, alpha=0.6, edgecolor=None)
    # Plot a diagonal line for reference
    max_val = max(max(true_values), max(predicted_values))
    min_val = min(min(true_values), min(predicted_values))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('True Bandgap (eV)', fontsize=14)
    plt.ylabel('Predicted Bandgap (eV)', fontsize=14)
    plt.title('True vs. Predicted Bandgap Values', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f'MAE_{mae:.4f}_MSE_{mse:.4f}.png'))  # Save the plot as a PNG file
    plt.close()
    print("Plot saved successfully.")

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
