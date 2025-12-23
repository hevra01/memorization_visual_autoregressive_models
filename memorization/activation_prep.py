#!/usr/bin/env python3
"""
Prepare per-scale transformer activations across the dataset for VAR.
- Hooks on MLP fc1, MLP fc2, and Self-Attention proj layers.
- Aggregates average activations over tokens for each scale.
- Saves results per head/module to separate output files.

Final saved tensor shapes: (num_blocks, dataset_size, num_scales, hidden_dim)
"""
import os
import sys
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(1, PROJECT_ROOT)

from models import build_vae_var
from memorization.data_prep.subset_imagenet import get_balanced_imagenet_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Prepare VAR activations for UnitMem")
    p.add_argument("--split", type=str, default="val_categorized", help="Dataset split (e.g., train, val, val_categorized)")
    p.add_argument("--batch_size", type=int, default=8, help="DataLoader batch size")
    p.add_argument("--output_dir", type=str, required=True, help="Directory to store activation files")
    p.add_argument("--model_depth", type=int, default=16, help="VAR depth (num blocks)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------
    # Build and load models
    # ---------------------------
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=args.device, patch_nums=patch_nums,
        num_classes=1000, depth=args.model_depth, shared_aln=False,
    )

    vae.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, "memorization/checkpoints/vae_ch160v4096z32.pth"),
        map_location="cpu"
    ))
    var.load_state_dict(torch.load(
        os.path.join(PROJECT_ROOT, f"memorization/checkpoints/var_d{args.model_depth}.pth"),
        map_location="cpu"
    ))

    vae.eval(); var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)

    # ---------------------------
    # Dataset
    # ---------------------------
    dataset = get_balanced_imagenet_dataset(split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    num_blocks = len(var.blocks)
    begin_ends = var.begin_ends  # token ranges per scale

    # ---------------------------
    # Storage: append per *batch*
    # ---------------------------
    per_block_fc1  = [[] for _ in range(num_blocks)]
    per_block_fc2  = [[] for _ in range(num_blocks)]
    per_block_attn = [[] for _ in range(num_blocks)]
    per_block_q    = [[] for _ in range(num_blocks)]
    per_block_k    = [[] for _ in range(num_blocks)]
    per_block_v    = [[] for _ in range(num_blocks)]

    # ---------------------------
    # Temporary hook buffers
    # ---------------------------
    fc1_batch_acts  = [[] for _ in range(num_blocks)]
    fc2_batch_acts  = [[] for _ in range(num_blocks)]
    attn_batch_acts = [[] for _ in range(num_blocks)]
    q_batch_acts    = [[] for _ in range(num_blocks)]
    k_batch_acts    = [[] for _ in range(num_blocks)]
    v_batch_acts    = [[] for _ in range(num_blocks)]

    # ---------------------------
    # Hooks
    # ---------------------------
    def make_fc1_hook(bi):
        return lambda m, i, o: fc1_batch_acts[bi].append(o.detach())

    def make_fc2_hook(bi):
        return lambda m, i, o: fc2_batch_acts[bi].append(o.detach())

    def make_attn_proj_hook(bi):
        return lambda m, i, o: attn_batch_acts[bi].append(o.detach())

    def make_attn_qkv_hook(bi):
        def hook(module, inp, out):
            # NOTE: SelfAttention.forward uses functional linear with custom bias, so hooking mat_qkv directly won't fire. 
            # Here we hook the SelfAttention module and recompute qkv from the input x and module parameters.
            x = inp[0]  # (B, L, C)
            bias = torch.cat((module.q_bias, module.zero_k_bias, module.v_bias))
            qkv = torch.nn.functional.linear(x, module.mat_qkv.weight, bias)
            B, L, _ = qkv.shape
            qkv = qkv.view(B, L, 3, module.num_heads, module.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q_batch_acts[bi].append(q.reshape(B, L, -1).detach())
            k_batch_acts[bi].append(k.reshape(B, L, -1).detach())
            v_batch_acts[bi].append(v.reshape(B, L, -1).detach())
        return hook

    # Registering the hooks: each block gets 4 hooks.
    # 2 hooks for the MLP (fc1, fc2) and 2 hooks for the Self-Attention (proj, qkv)
    handles = []
    for bi, blk in enumerate(var.blocks):
        handles += [
            blk.ffn.fc1.register_forward_hook(make_fc1_hook(bi)),
            blk.ffn.fc2.register_forward_hook(make_fc2_hook(bi)),
            blk.attn.proj.register_forward_hook(make_attn_proj_hook(bi)),
            blk.attn.register_forward_hook(make_attn_qkv_hook(bi)),
        ]

    # ---------------------------
    # Helper: token → scale aggregation
    # ---------------------------
    def aggregate_over_scales(x):
        """ 
            x is the activation tensor with shape (B, L, C), where B is batch size, L is number of tokens, and C is hidden dimension.
            The function aggregates activations over tokens corresponding to each scale, resulting in a tensor of shape (B, S, C), where S is the number of scales.
            The aggregation is done by averaging the activations over the tokens for each scale.
            
            x: (B, L, C) → (B, S, C). 
        """
        # x: (B, L, C) → (B, S, C)
        return torch.stack(
            [x[:, bg:ed].mean(dim=1) for (bg, ed) in begin_ends],
            dim=1
        )

    # ---------------------------
    # Main loop
    # ---------------------------
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(args.device)
            labels = labels.to(args.device)

            # Prepare VAE embeddings to be used in teacher-forcing, where ground truth is needed to be given as context.
            indices = var.vae_proxy[0].img_to_idxBl(images)
            emb = torch.cat(
                [var.vae_quant_proxy[0].embedding(idx) for idx in indices[1:]],
                dim=1
            )

            # Forward pass (teacher forcing), where the activations are captured by hooks.
            _ = var(label_B=labels, x_BLCv_wo_first_l=emb)

            for bi in range(num_blocks):
                per_block_fc1[bi].append(aggregate_over_scales(fc1_batch_acts[bi].pop(0)))
                per_block_fc2[bi].append(aggregate_over_scales(fc2_batch_acts[bi].pop(0)))
                per_block_attn[bi].append(aggregate_over_scales(attn_batch_acts[bi].pop(0)))
                per_block_q[bi].append(aggregate_over_scales(q_batch_acts[bi].pop(0)))
                per_block_k[bi].append(aggregate_over_scales(k_batch_acts[bi].pop(0)))
                per_block_v[bi].append(aggregate_over_scales(v_batch_acts[bi].pop(0)))

    # ---------------------------
    # Final stacking
    # ---------------------------
    fc1_all  = torch.stack([torch.cat(v, 0) for v in per_block_fc1], 0)
    fc2_all  = torch.stack([torch.cat(v, 0) for v in per_block_fc2], 0)
    attn_all = torch.stack([torch.cat(v, 0) for v in per_block_attn], 0)
    q_all    = torch.stack([torch.cat(v, 0) for v in per_block_q], 0)
    k_all    = torch.stack([torch.cat(v, 0) for v in per_block_k], 0)
    v_all    = torch.stack([torch.cat(v, 0) for v in per_block_v], 0)

    # ---------------------------
    # Save
    # ---------------------------
    torch.save(fc1_all,  os.path.join(args.output_dir, "activations_fc1.pt"))
    torch.save(fc2_all,  os.path.join(args.output_dir, "activations_fc2.pt"))
    torch.save(attn_all, os.path.join(args.output_dir, "activations_attn_proj.pt"))
    torch.save(q_all,    os.path.join(args.output_dir, "activations_q.pt"))
    torch.save(k_all,    os.path.join(args.output_dir, "activations_k.pt"))
    torch.save(v_all,    os.path.join(args.output_dir, "activations_v.pt"))

    for h in handles:
        h.remove()

if __name__ == "__main__":
    main()