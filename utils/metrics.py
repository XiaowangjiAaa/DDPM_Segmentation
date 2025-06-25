# utils/metrics.py
# 分割任务常用指标计算（支持二分类与多分类）：mIoU, accuracy, F1-score, recall, precision, dice

import torch
import torch.nn.functional as F

def one_hot(tensor, num_classes):
    """Convert [B, H, W] or [B, 1, H, W] to one-hot: [B, C, H, W]"""
    if tensor.ndim == 4 and tensor.size(1) == 1:
        tensor = tensor.squeeze(1)
    return F.one_hot(tensor.long(), num_classes).permute(0, 3, 1, 2).float()

def pixel_accuracy(pred_class, target):
    """Accuracy: pred_class and target should be [B, H, W]"""
    correct = (pred_class == target).float()
    return correct.sum() / correct.numel()

def dice_score(pred_onehot, target_onehot, epsilon=1e-6):
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean()

def iou_score(pred_onehot, target_onehot, epsilon=1e-6):
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = (pred_onehot + target_onehot - pred_onehot * target_onehot).sum(dim=(2, 3))
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean()

def precision_score(pred_onehot, target_onehot, epsilon=1e-6):
    tp = (pred_onehot * target_onehot).sum(dim=(2, 3))
    fp = (pred_onehot * (1 - target_onehot)).sum(dim=(2, 3))
    precision = (tp + epsilon) / (tp + fp + epsilon)
    return precision.mean()

def recall_score(pred_onehot, target_onehot, epsilon=1e-6):
    tp = (pred_onehot * target_onehot).sum(dim=(2, 3))
    fn = ((1 - pred_onehot) * target_onehot).sum(dim=(2, 3))
    recall = (tp + epsilon) / (tp + fn + epsilon)
    return recall.mean()

def f1_score(pred_onehot, target_onehot, epsilon=1e-6):
    prec = precision_score(pred_onehot, target_onehot, epsilon)
    rec = recall_score(pred_onehot, target_onehot, epsilon)
    return 2 * prec * rec / (prec + rec + epsilon)

def dice_loss(pred, target, smooth=1e-6):
    """用于训练阶段，输入为 logits 和 target（未 one-hot）"""
    pred_soft = F.softmax(pred, dim=1)[:, 1, :, :]  # 取前景类通道
    target = (target == 1).float()
    intersection = (pred_soft * target).sum()
    union = pred_soft.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice
