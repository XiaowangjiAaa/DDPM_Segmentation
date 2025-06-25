# diffusion/losses.py
# 常用扩散模型损失函数，如 MSE、L1、Dice 等
import torch.nn.functional as F

def diffusion_loss(pred, target, loss_type='mse'):
    if loss_type == 'mse':
        return F.mse_loss(pred, target)
    elif loss_type == 'l1':
        return F.l1_loss(pred, target)
    else:
        raise NotImplementedError(f"Unsupported loss type: {loss_type}")
