"""
Mean Activation Preparation
"""

import os
import torch
import wandb

def main():

    # Setup wandb
    # Initialize Weights & Biases for logging
    wandb.init(project="VaR_memorization", name="mean_activation_prep")

    augmented_activation_base_dir = "/scratch/inf0/user/hpetekka/var_mem/output_activations_corrected_abs/"
    mean_activation_dir = "/scratch/inf0/user/hpetekka/var_mem/output_activations_corrected_abs/mean"

    keys = [
        "fc1", "fc1_act", "fc2",
        "q", "k", "v", "attn_proj"
    ]
    keys = ["fc1_act",]

    for key in keys:

        # different blocks store their activations separately
        for j in range(16):

            # find mean activation over augmentations
            current_activation = []
            for i in range(10):
                current_file = os.path.join(augmented_activation_base_dir, str(i), key, f"block_{j}_combined.pt")
                current_activation.append(torch.load(current_file).float())
            
            shapes = [x.shape for x in current_activation]
            assert len(set(shapes)) == 1, f"Shape mismatch for {key}, block {j}: {shapes}"

            mean_activation = torch.mean(torch.stack(current_activation), dim=0)
            
            save_path = os.path.join(mean_activation_dir, key)
            os.makedirs(save_path, exist_ok=True)
            
            save_file = os.path.join(save_path, f"block_{j}.pt")
            torch.save(mean_activation, save_file)
            print(f"Saved mean activation for {key}, block {j} to {save_file}")


if __name__ == "__main__":
    main()