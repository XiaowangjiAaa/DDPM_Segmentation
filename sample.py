import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models.unet_model import UNetModel_newpreview
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.schedule import get_named_beta_schedule
from data.ISIC import ISICDataset
from sklearn.metrics import jaccard_score, f1_score

@torch.no_grad()
def main():
    ckpt_path = "results/best.pt"
    data_dir = "datasets/ISIC/Val"
    out_dir = "inference_outputs"
    os.makedirs(out_dir, exist_ok=True)

    image_size = 256
    num_samples = 1  # 每张图采样次数

    # === 模型定义 ===
    model = UNetModel_newpreview(
        image_size=image_size,
        in_channels=4,
        model_channels=64,
        out_channels=2,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        dropout=0.0,
        channel_mult=(1, 2, 4, 4),
        use_fp16=False,
        use_checkpoint=False,
        resblock_updown=False,
        use_scale_shift_norm=True,
        num_heads=1,
    ).cuda()

    # === 加载权重 ===
    ckpt = torch.load(ckpt_path, map_location="cuda")
    state_dict = ckpt.get("state_dict", ckpt)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # === Diffusion beta schedule ===
    betas = get_named_beta_schedule("linear", 1000)
    diffusion = GaussianDiffusion(betas)

    # === 数据加载 ===
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    dataset = ISICDataset(args=None, data_path=data_dir, transform=transform, mode="Val")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"[Sample] Loaded {len(dataset)} samples.")

    all_mious = []
    all_dices = []

    for batch in dataloader:
        x = batch["image"].cuda()                # [1, 4, H, W]
        gt = batch["mask"].squeeze(0).numpy()   # [H, W] GT 为 0/1
        name = batch["name"][0].split('.')[0]

        # Resize GT 为固定大小（如果没在 Dataset 中 resize）
        if gt.shape != (image_size, image_size):
            gt = np.array(Image.fromarray(gt).resize((image_size, image_size), resample=Image.NEAREST))

        sample_mious = []
        sample_dices = []

        for i in range(num_samples):
            t = torch.tensor([0], dtype=torch.long).cuda()
            out, _ = model(x, t)
            pred = torch.argmax(out, dim=1)[0].cpu().numpy()  # [H, W] 0/1 mask

            # 保存每次结果图（可选）
            out_path = os.path.join(out_dir, f"{name}_sample{i}.png")
            Image.fromarray((pred * 255).astype('uint8')).save(out_path)

            # 评估指标计算（flatten 二分类）
            miou = jaccard_score(gt.flatten(), pred.flatten(), average='binary')
            dice = f1_score(gt.flatten(), pred.flatten(), average='binary')

            sample_mious.append(miou)
            sample_dices.append(dice)

        # 输出该图平均
        avg_miou = np.mean(sample_mious)
        avg_dice = np.mean(sample_dices)
        all_mious.append(avg_miou)
        all_dices.append(avg_dice)

        print(f"[{name}] mIoU: {avg_miou:.4f}, Dice: {avg_dice:.4f}")

    print("===")
    print(f"[Overall] Avg mIoU: {np.mean(all_mious):.4f}, Avg Dice: {np.mean(all_dices):.4f}")

if __name__ == "__main__":
    main()
