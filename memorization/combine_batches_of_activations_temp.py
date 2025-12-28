"""
(activation_prep.py) script prepared the activations per batch, per block, per component due to limited storage.
This script combines these batches into single tensors per block, per component.
"""

import os
from pathlib import Path
import shutil
import torch
import wandb

def main():

    # Setup wandb
    # Initialize Weights & Biases for logging
    wandb.init(project="VaR_memorization", name="combine_batches_of_activations")
    
    main_dir = "/scratch/inf0/user/hpetekka/var_mem/output_activations_corrected_abs/"
    # "/scratch/inf0/user/hpetekka/var_mem/output_activations/0/attn_proj"
    # has block_0, block_1, ..., block_15 folders with attention projection activations saved as batch0.pt, batch1.pt, batch159.pt 
    # combine these into a single tensor per block.

    # loop over runs, which are for different augmented versions of the same data points.
    for i in range(10):
        # for each version, loop over activation types
        for dir in (
            #f"{main_dir}/{i}/fc1",
            f"{main_dir}/{i}/fc1_act",
            # f"{main_dir}/{i}/fc2",
            # f"{main_dir}/{i}/q",
            # f"{main_dir}/{i}/k",
            # f"{main_dir}/{i}/v",
            # f"{main_dir}/{i}/attn_proj"
        ):
            # for each activation type, loop over blocks, where each block corresponds to a scale.
            for block_i in range(0,16):
                block_folder = os.path.join(dir, f"block_{block_i}")
                all_batches = []
                for batch_i in range(116):
                    batch_file = os.path.join(block_folder, f"batch{batch_i}.pt")
                    batch_tensor = torch.load(batch_file)  # shape: (batch_size, num_tokens, hidden_dim)
                    all_batches.append(batch_tensor)
                # concatenate all batches
                block_tensor = torch.cat(all_batches, dim=0)  # shape: (num_data_points, num_tokens, hidden_dim)
                # save the combined tensor
                combined_file = os.path.join(dir, f"block_{block_i}_combined.pt")
                torch.save(block_tensor, combined_file)
                print(f"Saved combined activations for {dir}, block {block_i} to {combined_file}")

    
    base_dirs = [
    "fc1", "fc1_act", "fc2",
    "q", "k", "v", "attn_proj"
    ]

    base_dirs = ["fc1_act",]

    for i in range(0, 10):
        for subdir in base_dirs:
            dir_path = Path(f"{main_dir}/{i}/{subdir}")

            for p in dir_path.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)

if __name__ == "__main__":
    main()