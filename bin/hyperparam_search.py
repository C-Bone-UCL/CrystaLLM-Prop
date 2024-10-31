import subprocess
import yaml

# Define LoRA parameter configurations
lora_configs = [
    {"LoRA_rank": 16, "LoRA_alpha": 32},
    {"LoRA_rank": 32, "LoRA_alpha": 32},
    {"LoRA_rank": 32, "LoRA_alpha": 64},
    {"LoRA_rank": 8, "LoRA_alpha": 16},
]

# Path to the configuration file
config_file_path = "$HOME/CrystaLLM/config/finetune_LoRA_BG.yaml"

# SSH command template
command_template = """sudo apptainer exec --nv \
--bind /lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu \
--bind $HOME/CrystaLLM:/opt/CrystaLLM \
--pwd /opt/CrystaLLM $HOME/CrystaLLM/crystallm_container.sif \
python3 bin/train.py --config=config/finetune_LoRA_BG.yaml
"""

def update_config_file(config_path, lora_rank, lora_alpha):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the LoRA parameters and wandb_run_name
    config["LoRA_rank"] = lora_rank
    config["LoRA_alpha"] = lora_alpha
    config["wandb_run_name"] = f"BG_large_LoRA_{lora_rank}_{lora_alpha}"

    # Save the updated configuration
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

def run_ssh_command(command):
    # Run the SSH command in the terminal
    process = subprocess.run(command, shell=True)
    if process.returncode != 0:
        print(f"Command failed with exit code {process.returncode}")
    else:
        print("Command executed successfully")

# Loop through each configuration and execute the command
for config in lora_configs:
    # Update the config file
    update_config_file(config_file_path, config["LoRA_rank"], config["LoRA_alpha"])
    
    # Run the command with the updated configuration
    run_ssh_command(command_template)
