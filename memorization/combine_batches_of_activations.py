"""
(activation_prep.py) script prepared the activations per batch, per block due to limited storage.
This script combines these batches into single tensors per block, per component.
"""

import torch
from models import VQVAE, build_vae_var
from memorization.data_prep.subset_imagenet import get_balanced_imagenet_dataset
from pathlib import Path
import shutil
from glob import glob
import os


def combine_ranks_for_block(
    main_dir,
    layer,
    block_name,
    expected_num_images=12800,
):
    """
    Combine activation files from all ranks for one block and one layer.

    Produces a tensor of shape:
        (num_images, S, C)
    ordered by dataset index.
    """

    all_indices = []
    all_activations = []

    # Iterate over all rank directories (rank0, rank1, ...)
    for rank_dir in sorted(os.listdir(main_dir)):
        rank_path = os.path.join(main_dir, rank_dir, layer, block_name)

        if not os.path.isdir(rank_path):
            continue

        # Load all batch files from this rank
        batch_files = sorted(glob(os.path.join(rank_path, "batch*.pt")))

        for bf in batch_files:
            data = torch.load(bf, map_location="cpu")

            # data["indices"]: (B,)
            # data["activations"]: (B, S, C)
            all_indices.append(data["indices"])
            all_activations.append(data["activations"])

    # Concatenate everything
    all_indices = torch.cat(all_indices, dim=0)        # (N,)
    all_activations = torch.cat(all_activations, dim=0)  # (N, S, C)

    assert all_indices.shape[0] == all_activations.shape[0], \
        "Mismatch between indices and activations!"

    # Sort by dataset index
    sorted_indices, sort_order = torch.sort(all_indices)
    sorted_activations = all_activations[sort_order]

    # Sanity checks
    assert sorted_indices.unique().numel() == sorted_indices.numel(), \
        "Duplicate indices detected!"
    assert sorted_indices.numel() == expected_num_images, \
        f"Expected {expected_num_images} images, got {sorted_indices.numel()}"

    return sorted_activations, sorted_indices

def main():

    main_dir = "/scratch/inf0/user/hpetekka/var_mem/output_activations_corrected_test"
    layer = "fc1_act"

    final_dir = os.path.join(main_dir, "combined", layer)
    os.makedirs(final_dir, exist_ok=True)

    for block_id in range(16):
        block_name = f"block_{block_id}"

        activations, indices = combine_ranks_for_block(
            main_dir=main_dir,
            layer=layer,
            block_name=block_name,
            expected_num_images=12800,
        )

        # activations: (12800, 10, 4096)
        torch.save(
            {
                "activations": activations,
                "indices": indices,  # optional but nice to keep
            },
            os.path.join(final_dir, f"{block_name}.pt")
        )

        print(f"Saved combined {layer}/{block_name}: {activations.shape}")

if __name__ == "__main__":
    main()