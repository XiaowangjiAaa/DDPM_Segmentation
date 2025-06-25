import os
import torch as th
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributed import is_initialized, get_rank

from utils.metrics import dice_loss, dice_score, iou_score, f1_score
from utils import logger


def is_main_process():
    return not is_initialized() or get_rank() == 0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        dataloader,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint=None,
        use_fp16=False,
        weight_decay=0.0,
        logger=None,
        out_dir="checkpoints",
        val_loader=None,
        val_interval=1000,
        use_scheduler=True,
        lr_scheduler_tmax=200000,
        train_steps=200000,
    ):
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader
        self.val_loader = val_loader
        self.val_interval = val_interval

        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = float(ema_rate)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.weight_decay = weight_decay
        self.logger = logger
        self.out_dir = out_dir
        self.train_steps = train_steps

        self.step = 0
        self.best_dice = 0.0
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if use_scheduler:
            self.scheduler = CosineAnnealingLR(self.opt, T_max=lr_scheduler_tmax, eta_min=1e-6)
        else:
            self.scheduler = None

    def run_loop(self):
        self.model.train()
        while self.step < self.train_steps:
            for batch in tqdm(self.dataloader):
                loss = self.run_step(batch)
                self.step += 1

                if self.step % self.log_interval == 0 and is_main_process():
                    print(f"[Step {self.step}] step logged.")
                    if self.logger:
                        self.logger.log_metrics({"step": self.step}, step=self.step)

                if self.step % self.save_interval == 0 and is_main_process():
                    self.save_model(self.step, loss)

                if self.val_loader and self.step % self.val_interval == 0:
                    self.run_validation(self.val_loader)

                if self.step >= self.train_steps:
                    break

    def run_step(self, batch):
        self.opt.zero_grad()

        x = batch["image"].cuda()
        y = batch["mask"].long().squeeze(1).cuda()
        t = th.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=x.device)

        out, _ = self.model(x, t)
        ce = F.cross_entropy(out, y)
        dice = dice_loss(out, y)
        loss = ce + 0.5 * dice

        loss.backward()
        self.opt.step()
        if self.scheduler:
            self.scheduler.step()

        if self.step % 10 == 0 and is_main_process():
            print(f"[step {self.step}] seg_loss: {loss.item():.4f}")
            print(f"[{self.step}] CE: {ce.item():.4f}, Dice: {dice.item():.4f}")
            if self.logger:
                self.logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/ce": ce.item(),
                    "train/dice": dice.item(),
                    "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.lr
                }, step=self.step)

        return loss.item()

    def run_validation(self, val_loader):
        self.model.eval()
        model = self.model.module if hasattr(self.model, "module") else self.model
        dices, ious, f1s = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].cuda()
                y = batch["mask"].long().squeeze(1).cuda()
                t = th.zeros(x.size(0), dtype=torch.long).cuda()

                out, _ = model(x, t)
                pred_class = out.argmax(dim=1)

                pred_onehot = F.one_hot(pred_class, num_classes=2).permute(0, 3, 1, 2).float()
                target_onehot = F.one_hot(y, num_classes=2).permute(0, 3, 1, 2).float()

                dices.append(dice_score(pred_onehot, target_onehot).item())
                ious.append(iou_score(pred_onehot, target_onehot).item())
                f1s.append(f1_score(pred_onehot, target_onehot).item())

                if self.logger and is_main_process() and len(dices) <= 4:
                    for i in range(x.shape[0]):
                        rgb = (x[i, :3].detach().cpu().numpy() * 255).astype(np.uint8)
                        rgb = np.transpose(rgb, (1, 2, 0))
                        pred_vis = (pred_class[i].detach().cpu().numpy() * 255).astype(np.uint8)
                        gt_vis = (y[i].detach().cpu().numpy() * 255).astype(np.uint8)

                        images = [Image.fromarray(rgb), Image.fromarray(gt_vis), Image.fromarray(pred_vis)]
                        names = [f"{i}_input", f"{i}_gt", f"{i}_pred"]
                        self.logger.log_images(images=images, names=names, step=self.step)

        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(ious) / len(ious)
        avg_f1 = sum(f1s) / len(f1s)

        if is_main_process():
            print(f"[Validation] Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f} | F1: {avg_f1:.4f}")
            if self.logger:
                self.logger.log_metrics({
                    "val/dice": avg_dice,
                    "val/iou": avg_iou,
                    "val/f1": avg_f1
                }, step=self.step)

            if avg_dice > self.best_dice:
                self.best_dice = avg_dice
                best_path = os.path.join(self.out_dir, "best.pt")
                th.save(self.model.state_dict(), best_path)
                print(f"[Best Model] Dice improved to {avg_dice:.4f}, saved to {best_path}")

        self.model.train()

    def save_model(self, step, loss_value):
        os.makedirs(self.out_dir, exist_ok=True)
        loss_str = f"{loss_value:.4f}"
        path = os.path.join(self.out_dir, f"model_step{step}_loss{loss_str}.pt")
        th.save(self.model.state_dict(), path)
        print(f"[Checkpoint] Saved model to {path}")
