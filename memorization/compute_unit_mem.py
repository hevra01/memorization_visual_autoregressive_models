import os
import torch

def compute_unitmem(mean_max_activations_dict, eps=1e-8):
    """
    Compute UnitMem scores from precomputed μ_max and μ_-max values.

    Parameters
    ----------
    mean_max_activations_dict : dict
        Nested structure:
        mean_max_activations_dict[key][block][scale] -> tensor of shape (C, 2),
        where:
            [:, 0] = μ_max,u      (maximum mean activation across data points)
            [:, 1] = μ_-max,u     (mean activation over remaining data points)

    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    unitmem_dict : dict
        Same structure as input, but:
        unitmem_dict[key][block][scale] -> tensor of shape (C,)
        containing UnitMem scores per unit.
    """

    unitmem_dict = {}

    for key in mean_max_activations_dict:
        unitmem_dict[key] = []

        for block in range(len(mean_max_activations_dict[key])):
            unitmem_dict[key].append([])

            for scale in range(len(mean_max_activations_dict[key][block])):
                # tensor shape: (C, 2)
                mean_max_tensor = mean_max_activations_dict[key][block][scale]

                # split μ_max,u and μ_-max,u
                mu_max = mean_max_tensor[:, 0]      # (C,)
                mu_minus_max = mean_max_tensor[:, 1]  # (C,)

                # UnitMem_D'(u) = (μ_max,u - μ_-max,u) / (μ_max,u + μ_-max,u)
                unitmem = (mu_max - mu_minus_max) / (mu_max + mu_minus_max + eps)

                unitmem_dict[key][block].append(unitmem)

    return unitmem_dict

def main():
    mean_activation_over_augmentations_dir = (
        "/scratch/inf0/user/hpetekka/var_mem/output_activations_corrected_abs/mean"
    )

    keys = ["fc1", "fc1_act", "fc2", "q", "k", "v", "attn_proj"]

    # Structure:
    # mean_max_activations_dict[key][block][scale] -> (C, 2) [max, mean]
    mean_max_activations_dict = {
        key: [ [None for _ in range(10)] for _ in range(16) ]
        for key in keys
    }

    for key in keys:
        for block in range(16):
            current_file = os.path.join(
                mean_activation_over_augmentations_dir,
                key,
                f"block_{block}.pt"
            )

            # shape: (N=12800, S=10, C)
            current_activation = torch.load(current_file).float()

            for scale in range(10):
                scale_act = current_activation[:, scale, :] # (N, C)
                N = scale_act.shape[0]

                # μ_max,u and argmax index k (per unit)
                mu_max, k = scale_act.max(dim=0)   # both shape (C,)

                # sum over all data points (per unit)
                sum_all = scale_act.sum(dim=0)     # (C,)

                # μ_-max,u = (sum_all - μ_max) / (N - 1)
                mu_minus_max = (sum_all - mu_max) / (N - 1)

                # store as (C, 2)
                mean_max_activations_dict[key][block][scale] = torch.stack(
                    [mu_max, mu_minus_max],
                    dim=1
                )

   
   # Once we have mean_max_activations_dict, we can compute UnitMem scores.
    unitmem_dict = compute_unitmem(
        mean_max_activations_dict,
        eps=1e-8
    )

    torch.save(unitmem_dict, "unitmem_scores_corrected_abs.pt")
    print("Saved UnitMem scores to unitmem_scores_corrected_abs.pt")
    
if __name__ == "__main__":
    main()