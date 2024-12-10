import subprocess
import os
from ruamel.yaml import YAML

# Initialize ruamel.yaml
yaml = YAML()
yaml.preserve_quotes = True  # Preserve quotes if any

# Define LoRA parameter configurations
lora_configs = [
    {"LoRA_rank": 2, "LoRA_alpha": 4},
]

# Parameters to set for all the config files
configs = [
    {"beta2": 0.995, "weight_decay": 0.1, "dropout": 0.1},
]

# List of configuration files to cycle through
config_files = [
    "config/cif_extd_BG/finetune_LoRA_BG_nico.yaml",
    "config/cif_extd_BG/finetune_head_BG_nico.yaml",
    "config/regression_BG/regression_BG_all_nico.yaml",
    "config/regression_BG/regression_BG_head_nico.yaml",
    "config/regression_BG/regression_BG_LoRA_myriad.yaml",
    # "config/finetune_all_BG_nico.yaml",
]

# Command template with a placeholder for the config file
command_template = "python3 bin/train.py --config={config_file}"

def update_nested_config(config, updates):
    """
    Recursively updates a dictionary (config) with values from another dictionary (updates).
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in config and isinstance(config[key], dict):
            update_nested_config(config[key], value)
        else:
            config[key] = value

def update_config_file(config_path, **kwargs):
    # Check if the config file exists
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        return False

    # Load the existing configuration with ruamel.yaml to preserve structure
    with open(config_path, 'r') as file:
        config = yaml.load(file)

    # Update the configuration with the new parameters
    update_nested_config(config, kwargs)

    # Set wandb_run_name based on the config file name without extension
    config["wandb_run_name"] = os.path.basename(config_path).split(".")[0]
    run_name = os.path.basename(config_path).split(".")[0]

    print(f"wand run name: {run_name}")

    # Save the updated configuration, preserving comments and order
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

    return True

def run_command(command):
    print(f"Running command: {command}")  # Log the command for debugging
    try:
        # Run without capturing output to see real-time command progress
        process = subprocess.run(command, shell=True, check=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(e)  # Print full error details for debugging

# Loop through each configuration file
for config_file in config_files:
    print(f"Processing configuration file: {config_file}")
    if "LoRA" in os.path.basename(config_file):
        # Apply both LoRA and general configurations
        for config in configs:
            for lora_config in lora_configs:
                combined_config = {**lora_config, **config}  # Merge dictionaries
                print(f"Applying combined LoRA and general config to {config_file}: {combined_config}")
                
                # Update the configuration file
                success = update_config_file(config_file, **combined_config)
                if not success:
                    continue  # Skip if config update failed

                # Run the command
                command = command_template.format(config_file=config_file)
                run_command(command)
    else:
        # Apply only general configurations
        for config in configs:
            print(f"Applying general config to {config_file}: {config}")
            
            # Update the configuration file
            success = update_config_file(config_file, **config)
            if not success:
                continue  # Skip if config update failed

            # Run the command
            command = command_template.format(config_file=config_file)
            run_command(command)
