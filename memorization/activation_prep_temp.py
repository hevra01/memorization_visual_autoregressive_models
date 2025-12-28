"""
1) Save the activations of various components of VAR model: fc1, fc1_act, fc2, attn_proj, q, k, v.
2) This file is run as a Slurm job array, where each job processes the entire subset (12800 samples, 1% of imagenet trainset) but with 
   different augmentations (10 augmentations total). 
3) It saves the activations per batch, per block, per component due to limited storage.
4) Later, another script (combine_batches_of_activations.py) will combine the batches altogether.
"""
import os
import sys
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
import wandb
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

from models import build_vae_var
from memorization.data_prep.subset_imagenet import get_balanced_subset
from utils.data import normalize_01_into_pm1
from utils.data_sampler import EvalDistributedSampler
import dist as dist  # repo's distributed utility helpers

def model_size_gb(model):
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 3)

def parse_args():
    p = argparse.ArgumentParser(description="Prepare VAR activations for UnitMem")
    p.add_argument("--split", type=str, default="train", help="Dataset split (e.g., train, val, val_categorized)")
    # Batch size is per-GPU/process. Global batch = batch_size * world_size.
    p.add_argument("--batch_size", type=int, default=140, help="Per-GPU DataLoader batch size")
    p.add_argument("--output_dir", type=str, default="/scratch/inf0/user/hpetekka/var_mem/output_activations_corrected_test/", help="Directory to store activation files")
    p.add_argument("--model_depth", type=int, default=16, help="VAR depth (num blocks)")
    # Device will be overridden by dist.get_device() when distributed is initialized.
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # New args: augmentation control + dataset root
    p.add_argument("--num_augmentations", type=int, default=10, help="Number of random augmentations per image batch")
    p.add_argument("--final_reso", type=int, default=256, help="Final resolution for crops (HxW)")
    p.add_argument("--mid_reso", type=float, default=1.125, help="Resize shorter edge to round(mid_reso*final_reso) before cropping")
    p.add_argument("--imagenet_root", type=str, default="/scratch/inf0/user/mparcham/ILSVRC2012", help="ImageNet root directory")
    p.add_argument("--total_samples", type=int, default=12800, help="Total dataset samples for balanced subset")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------------
    # Initialize (optional) distributed environment using repo helper
    # - If launched with torchrun and RANK env set, enables multi-GPU.
    # - Otherwise, runs single-GPU/CPU.
    # --------------------------------------------------------------
    dist.initialize(fork=False)
    args.device = dist.get_device() if torch.cuda.is_available() else args.device

    # Setup wandb (log only on master to avoid duplicate runs)
    if dist.is_master():
        wandb.init(project="VaR_memorization", name=f"activation_prep_run")
        wandb.config.update({**vars(args), "world_size": dist.get_world_size(), "rank": dist.get_rank()})
    else:
        # Prevent non-master ranks from creating separate W&B runs
        os.environ.setdefault("WANDB_MODE", "disabled")

    # ---------------------------
    # Build and load models
    # ---------------------------
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=args.device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )

    # the attention mask has been tailored according to unit_mem attention.
    # HENCE, remove it from the saved checkpoint, so that it doesn't override the correct one.
    ckpt = torch.load(os.path.join(PROJECT_ROOT, f"memorization/checkpoints/var_d{args.model_depth}.pth"), map_location='cpu')
    ckpt.pop('attn_bias_for_masking', None)
    var.load_state_dict(ckpt, strict=False)

    vae.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "memorization/checkpoints/vae_ch160v4096z32.pth"),
        map_location="cpu"
    ))

    vae.eval(); var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)

    # Model summary
    total_params_var = sum(p.numel() for p in var.parameters())
    total_params_vae = sum(p.numel() for p in vae.parameters())
    vae_model_size_gb = model_size_gb(vae)
    var_model_size_gb = model_size_gb(var)
    total_model_size_gb = vae_model_size_gb + var_model_size_gb

    if dist.is_master():
        print(f"VAE model has {total_params_vae/1e6:.2f}M parameters.")
        print(f"VAR model with depth {args.model_depth} has {total_params_var/1e6:.2f}M parameters.")
        print(f"VAE model size: {vae_model_size_gb:.2f} GB")
        print(f"VAR model size: {var_model_size_gb:.2f} GB")
        print(f"Total model size: {total_model_size_gb:.2f} GB")

    # ---------------------------
    # Dataset (no transform at dataset level)
    # We move all random transforms into this script to:
    #  - run multiple augmentations per batch
    #  - average activations across augmentations
    # The ImageFolder returns PIL images; we use a custom collate_fn to keep
    # them as a list and will transform inside the training loop.
    # ---------------------------
    # --------------------------------------------------------------
    # Dataset (rank-sharded via EvalDistributedSampler)
    # - Each rank processes a disjoint subset for larger global batch.
    # - Collate keeps PIL images; we transform inside the loop.
    # --------------------------------------------------------------

    # ---------------------------
    # Dataset with index tracking
    # ---------------------------
    class IndexedDataset(torch.utils.data.Dataset):
        """
        Wraps an existing dataset so that __getitem__ returns (img, label, index).
        The index refers to the position in the wrapped dataset (after subsetting).
        """
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            return img, label, idx


    base_dataset = ImageFolder(
        root=os.path.join(args.imagenet_root, args.split),
        transform=None,
    )
    dataset = get_balanced_subset(
        dataset=base_dataset,
        total_samples=args.total_samples,
        shuffle=True,
        seed=0,
    )

    # Wrap to expose indices
    dataset = IndexedDataset(dataset)

    # ---------------------------
    # Collate function (keeps PIL + index)
    # ---------------------------
    def collate_pil(batch):
        """
        batch: List[(PIL.Image, int label, int index)]
        returns:
            images: List[PIL.Image]
            labels: LongTensor (B,)
            indices: LongTensor (B,)  # stable dataset indices
        """
        imgs, labels, indices = zip(*batch)
        return (
            list(imgs),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(indices, dtype=torch.long),
        )

    # Use distributed sampler to split dataset per-rank. If not initialized, it
    # gracefully becomes a single-process full-range sampler.
    sampler = EvalDistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False, collate_fn=collate_pil)

    num_blocks = len(var.blocks)
    begin_ends = var.begin_ends  # token ranges per scale

    # ---------------------------
    # Temporary hook buffers:
    # We collect activations per block per call to `var()`. With multiple
    # augmentations, each block accumulates `num_augmentations` tensors which
    # we then mean-reduce (and take abs) before saving.
    # ---------------------------
    # Per-block activation buffers aggregate across augmentations:
    fc1_act_batch_acts  = [[] for _ in range(num_blocks)]
    
    
    def make_fc1_act_hook(bi):
        # lambda is a function that takes (module, input, output) and appends output.detach() to a list.
        # note: activations can stay in CPU.
        return lambda module, input, output: fc1_act_batch_acts[bi].append(output.detach().to(torch.float16).cpu())

    

    # Registering the hooks: each block gets 4 hooks.
    # 2 hooks for the MLP (fc1, fc2) and 2 hooks for the Self-Attention (proj, qkv)
    # Register hooks on each block's `ffn.act` to capture fc1 activation.
    handles = []
    for bi, blk in enumerate(var.blocks):
        handles += [
            blk.ffn.act.register_forward_hook(make_fc1_act_hook(bi)),
            
        ]

    # ---------------------------
    # Helper: token → scale aggregation
    # ---------------------------
    def aggregate_over_scales_ABLC(x, begin_ends):
        """
        Aggregate activations over tokens belonging to each scale.

        Parameters
        ----------
        x : torch.Tensor
            Shape (A, B, L, C)
            A = #augmentations
            B = batch size
            L = number of tokens
            C = hidden dim

        begin_ends : list of (int, int)
            Token ranges for each scale.

        Returns
        -------
        torch.Tensor
            Shape (A, B, S, C)
            S = number of scales
        """
        # For each scale, average over its token range
        # result per scale: (A, B, C)
        per_scale = [
            x[:, :, bg:ed, :].mean(dim=2)
            for (bg, ed) in begin_ends
        ]

        # Stack along scale dimension
        # (A, B, S, C)
        return torch.stack(per_scale, dim=2)
    
    
    # Use rank-specific subfolders to avoid filename collisions across ranks.
    rank_tag = f"rank{dist.get_rank()}" if dist.initialized() else "rank0"
    fc1_act_folder = os.path.join(args.output_dir, rank_tag, "fc1_act")
    os.makedirs(fc1_act_folder, exist_ok=True)

    
    # create sub-folders for each block
    for bi in range(num_blocks):
        os.makedirs(os.path.join(fc1_act_folder, f"block_{bi}"), exist_ok=True)
        
    # ---------------------------
    # Augmentation pipeline (moved from dataset into script)
    # This matches the VAR preprocessing:
    #   - Resize the shorter edge to round(mid_reso*final_reso) with LANCZOS
    #   - RandomCrop to (final_reso, final_reso)
    #   - ToTensor -> normalize from [0,1] into [-1,1]
    # We'll apply this `args.num_augmentations` times per batch and average.
    # ---------------------------
    mid_px = round(args.mid_reso * args.final_reso)
    aug_transform = T.Compose([
        T.Resize(mid_px, interpolation=InterpolationMode.LANCZOS),
        T.RandomCrop((args.final_reso, args.final_reso)),
        T.ToTensor(),
        normalize_01_into_pm1,
    ])

    # ---------------------------
    # Main loop
    # ---------------------------
    # --------------------------------------------------------------
    # Main loop: multi-augmentation inference on each rank's shard.
    # - Each rank saves its own outputs under {run}/{rank}/fc1_act/...
    # --------------------------------------------------------------
    processed_batches_local = 0  # counts local batches completed by this rank/process
    with torch.no_grad():
        for batch_idx, (images_list, labels, img_indices) in enumerate(loader):
            # images_list: List[PIL.Image]; labels: Tensor[long]
            # Run multiple augmentations, collect activations via hooks.
            for aug_i in range(args.num_augmentations):
                # Apply VAR-style augmentation to each image in the batch
                images_aug = torch.stack([aug_transform(img) for img in images_list], dim=0).to(args.device)
                labels_dev = labels.to(args.device)

                # Prepare VAE embeddings for teacher-forcing
                indices = var.vae_proxy[0].img_to_idxBl(images_aug)
                emb = torch.cat(
                    [var.vae_quant_proxy[0].embedding(idx) for idx in indices[1:]],
                    dim=1
                )

                # Forward pass (teacher forcing): hooks record activations
                _ = var(label_B=labels_dev, x_BLCv_wo_first_l=emb)
                if dist.is_master():
                    print(len(fc1_act_batch_acts[0]), fc1_act_batch_acts[0][0].shape)
                    print(fc1_act_batch_acts[0][0].device)  # Debug: check hook activations
                
            # After num_augmentations forward passes, average activations over augmentations
            # and take absolute value before aggregating over scales and saving.
            for bi in range(num_blocks):
                # fc1_act_batch_acts[bi]: List[num_augmentations x Tensor(B,L,C)] on CPU
                if len(fc1_act_batch_acts[bi]) == 0:
                    continue

                # Stack along augmentation dimension, compute mean in float32 for stability
                # change from fc1_act_batch_acts[bi]: List[num_augmentations x Tensor(B,L,C)] to (num_augmentations, B, L, C)
                acts_aug = torch.stack(fc1_act_batch_acts[bi], dim=0)  
                if dist.is_master():
                    print(acts_aug.shape)  # Debug: check shape after stacking
                acts_abs = acts_aug.abs() # take the absolute value    
                mean_over_scale = aggregate_over_scales_ABLC(acts_abs, begin_ends)
                # mean over augmentations
                mean_aug = mean_over_scale.mean(dim=0)   # (B, S, C)

                # Aggregate token activations into per-scale activations and save
                torch.save(
                    {
                        "indices": img_indices.cpu(),
                        "activations": mean_aug,
                    },
                    
                    f"{fc1_act_folder}/block_{bi}/batch{batch_idx}.pt"
                )

                # Clear buffer for next batch
                fc1_act_batch_acts[bi].clear()

            # ----------------------------------------------
            # Progress prints
            # - Per-rank local progress (printed by all ranks)
            # - Global aggregated number of processed batches (printed by master)
            # ----------------------------------------------
            processed_batches_local += 1
            print(f"[rk={dist.get_rank()}] local batch done: {processed_batches_local}")
            if dist.initialized():
                t_batches = torch.tensor(float(processed_batches_local))
                dist.allreduce(t_batches)  # sum across ranks
                global_processed = int(t_batches.item())
                if dist.is_master():
                    print(f"[global] processed_batches={global_processed} (world_size={dist.get_world_size()})")

    # Ensure all ranks have finished saving before cleaning up hooks.
    dist.barrier()
    for h in handles:
        h.remove()

if __name__ == "__main__":
    main()