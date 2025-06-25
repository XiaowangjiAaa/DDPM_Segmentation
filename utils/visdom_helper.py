# utils/visdom_helper.py
# 用于替代 visdom：可选接入 wandb 图像可视化支持

import wandb
import numpy as np
import torch
import torchvision.utils as vutils



def visualize_batch(inputs, outputs, targets, step=0, max_images=4, caption="sample"):
    """
    上传图像到 wandb：输入图像 + mask 预测 + mask GT
    inputs: [B, C, H, W] tensor
    outputs: [B, C, H, W] tensor (logits)
    targets: [B, H, W] or [B, 1, H, W] tensor
    """
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().float()
    if isinstance(outputs, torch.Tensor):
        outputs = torch.softmax(outputs.detach().cpu(), dim=1)
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu()
        if targets.ndim == 4:
            targets = targets.squeeze(1)

    B = min(inputs.size(0), max_images)
    viz_images = []
    for i in range(B):
        inp = inputs[i]
        pred_mask = outputs[i].argmax(0).float() / outputs.shape[1]  # normalize
        gt_mask = targets[i].float() / outputs.shape[1]

        combined = torch.stack([
            inp[0], pred_mask, gt_mask  # 显示单通道图/预测/标签
        ], dim=0)
        grid = vutils.make_grid(combined.unsqueeze(1), nrow=1, normalize=True, scale_each=True)
        viz_images.append(wandb.Image(grid, caption=f"{caption}_{i}"))

    wandb.log({f"{caption}_samples": viz_images}, step=step)