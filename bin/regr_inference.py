import os
import gzip
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from omegaconf import OmegaConf

from bin.train import TrainDefaults
from crystallm._model import GPT_regression, GPTConfig
from crystallm import CIFTokenizer, parse_config

# Configuration
CHECKPOINT_PATH = 'model_ckpts/regression_models/BG_head_test/ckpt.pt'  #
TEST_DATA_PATH = 'CIF_BG_proj/test_dataset.pkl.gz'
CONFIG_PATH = 'config/regression_BG/regression_BG_head.yaml' 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Dataset class
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
        if len(x) < self.block_size:
            x = x + [0] * (self.block_size - len(x))
        else:
            x = x[:self.block_size]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(self.data.loc[idx, 'Bandgap (eV)'], dtype=torch.float32)
        return x, y

def load_test_data(test_data_path):
    with gzip.open(test_data_path, 'rb') as f:
        test_df = pickle.load(f)
    print(f"Loaded test dataset with {len(test_df)} samples.")
    return test_df

def initialize_model(checkpoint_path, config):
    model = GPT_regression(config)
    # Load the checkpoint dictionary
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    # Check if 'model' key exists in the checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        # remove _orig_mod. from the keys of the state_dict
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    else:
        raise KeyError("Checkpoint does not contain 'model' key. Ensure the checkpoint format is correct.")
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    # Move the model to the specified device and set to evaluation mode
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {checkpoint_path} and moved to {DEVICE}.")
    return model


def perform_inference(model, test_loader):
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds, _ = model(inputs)
            preds = preds.squeeze(-1)  # Shape: (batch_size,)
            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(targets.cpu().numpy())
    print("Inference completed on test dataset.")
    # print stats
    all_trues = [round(val, 3) for val in all_trues]
    all_preds = [round(val, 3) for val in all_preds]
    print(f"True values: {all_trues[:5]} ...")
    print(f"Predicted values: {all_preds[:5]} ...")
    # print mse, r1, mae
    mse = torch.nn.functional.mse_loss(torch.tensor(all_trues), torch.tensor(all_preds)).item()
    r1 = torch.nn.functional.l1_loss(torch.tensor(all_trues), torch.tensor(all_preds)).item()
    mae = torch.nn.functional.l1_loss(torch.tensor(all_trues), torch.tensor(all_preds)).item()
    print(f"MSE: {mse:.3f}, R1: {r1:.3f}, MAE: {mae:.3f}")
    return all_trues, all_preds

def plot_true_vs_predicted(true_values, predicted_values):
    """
    Plot a scatter plot of true vs. predicted BG values.

    :param true_values: List or array of true BG values
    :param predicted_values: List or array of predicted BG values
    """
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
    plt.savefig('true_vs_predicted_bandgap.png')  # Save the plot as a PNG file
    plt.close()  # Close the figure to free memory
    print("Plot displayed successfully.")

def main():
    # Load the test dataset
    test_df = load_test_data(TEST_DATA_PATH)

    # Initialize the model configuration from TrainDefaults, then update with the config file using omegaconf
    config_dict = parse_config(TrainDefaults)
    # config_dict_yaml = OmegaConf.to_yaml(config_dict)
    with open(CONFIG_PATH, 'r') as f:
        config_dict.update(yaml.safe_load(f))

    print(f"Model configuration: {config_dict}")

    model_args = dict(
        n_layer=config_dict.n_layer,
        n_head=config_dict.n_head,
        n_embd=config_dict.n_embd,
        block_size=config_dict.block_size,
        bias=config_dict.bias,
        vocab_size=371,
        dropout=config_dict.dropout,
    )
    config = GPTConfig(**model_args)

    # Initialize the Dataset and DataLoader
    test_dataset = CIFRegressionDataset(test_df, block_size=config_dict.block_size)
    test_loader = DataLoader(test_dataset, batch_size=config_dict.batch_size, shuffle=False)
    print(f"Test DataLoader created with batch size {config_dict.batch_size}.")

    # Determine vocab_size from the dataset
    all_tokens = set(token for tokens in test_df['CIFs_tokenized'] for token in tokens)
    config.vocab_size = max(len(all_tokens), 371)  # Ensure vocab_size is at least 371
    print(f"Vocabulary size determined: {config.vocab_size} tokens.")

    # Initialize the model and load trained weights
    model = initialize_model(CHECKPOINT_PATH, config)

    # Perform inference
    true_vals, pred_vals = perform_inference(model, test_loader)

    # Plot the results
    plot_true_vs_predicted(true_vals, pred_vals)

def run_inference():
    main()

if __name__ == "__main__":
    run_inference()
