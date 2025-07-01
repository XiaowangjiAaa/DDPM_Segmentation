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
from accelerate import Accelerator

from utils.metrics import dice_score, iou_score, f1_score
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
        accelerator: Accelerator = None,
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
        self.accelerator = accelerator

        self.step = 0
        self.best_dice = 0.0
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if use_scheduler:
            self.scheduler = CosineAnnealingLR(self.opt, T_max=lr_scheduler_tmax, eta_min=1e-6)
        else:
            self.scheduler = None

        if self.accelerator is not None:
            if self.val_loader is not None:
                self.model, self.opt, self.dataloader, self.val_loader = self.accelerator.prepare(
                    self.model, self.opt, self.dataloader, self.val_loader
                )
            else:
                self.model, self.opt, self.dataloader = self.accelerator.prepare(
                    self.model, self.opt, self.dataloader
                )

    def run_loop(self):
        self.model.train()
        while self.step < self.train_steps:
            for batch in tqdm(self.dataloader):
                loss = self.run_step(batch)
                self.step += 1

                if self.step % self.log_interval == 0 and (
                    self.accelerator.is_main_process if self.accelerator else is_main_process()
                ):
                    print(f"[Step {self.step}] step logged.")
                    if self.logger:
                        self.logger.log_metrics({"step": self.step}, step=self.step)

                if self.step % self.save_interval == 0 and (
                    self.accelerator.is_main_process if self.accelerator else is_main_process()
                ):
                    self.save_model(self.step, loss)

                if self.val_loader and self.step % self.val_interval == 0:
                    self.run_validation(self.val_loader)

                if self.step >= self.train_steps:
                    break

    def run_step(self, batch):
        self.opt.zero_grad()

        device = self.accelerator.device if self.accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = batch["image"].to(device)
        y = batch["mask"].float().to(device)
        t = th.randint(0, self.diffusion.num_timesteps, (x.size(0),), device=device)

        loss = self.diffusion.p_losses(self.model, y, t, cond=x)

        if self.accelerator:
            self.accelerator.backward(loss)
        else:
            loss.backward()
        self.opt.step()
        if self.scheduler:
            self.scheduler.step()

        if self.step % 10 == 0 and (
            self.accelerator.is_main_process if self.accelerator else is_main_process()
        ):
            print(f"[step {self.step}] loss: {loss.item():.4f}")
            if self.logger:
                self.logger.log_metrics({
                    "train/loss": loss.item(),
                    "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.lr
                }, step=self.step)

        return loss.item()

    def run_validation(self, val_loader):
        self.model.eval()
        model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator
            else (self.model.module if hasattr(self.model, "module") else self.model)
        )
        dices, ious, f1s = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                device = self.accelerator.device if self.accelerator else torch.device("cuda" if torch.cuda.is_available() else "cpu")
                x = batch["image"].to(device)
                y = batch["mask"].float().to(device)

                pred = self.diffusion.p_sample_loop(model, y.shape, cond=x)
                pred_class = (pred > 0.5).long().squeeze(1)

                pred_onehot = F.one_hot(pred_class, num_classes=2).permute(0, 3, 1, 2).float()
                target_onehot = F.one_hot(y.long().squeeze(1), num_classes=2).permute(0, 3, 1, 2).float()

                dices.append(dice_score(pred_onehot, target_onehot).item())
                ious.append(iou_score(pred_onehot, target_onehot).item())
                f1s.append(f1_score(pred_onehot, target_onehot).item())

                if self.logger and (
                    self.accelerator.is_main_process if self.accelerator else is_main_process()
                ) and len(dices) <= 4:
                    for i in range(x.shape[0]):
                        rgb = (x[i, :3].detach().cpu().numpy() * 255).astype(np.uint8)
                        rgb = np.transpose(rgb, (1, 2, 0))
                        pred_vis = (pred_class[i].detach().cpu().numpy() * 255).astype(np.uint8)
                        gt_vis = (y[i, 0].detach().cpu().numpy() * 255).astype(np.uint8)

                        images = [Image.fromarray(rgb), Image.fromarray(gt_vis), Image.fromarray(pred_vis)]
                        names = [f"{i}_input", f"{i}_gt", f"{i}_pred"]
                        self.logger.log_images(images=images, names=names, step=self.step)

        avg_dice = sum(dices) / len(dices)
        avg_iou = sum(ious) / len(ious)
        avg_f1 = sum(f1s) / len(f1s)

        if self.accelerator.is_main_process if self.accelerator else is_main_process():
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
                to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
                if self.accelerator:
                    self.accelerator.wait_for_everyone()
                th.save(to_save.state_dict(), best_path)
                print(f"[Best Model] Dice improved to {avg_dice:.4f}, saved to {best_path}")

        self.model.train()

    def save_model(self, step, loss_value):
        os.makedirs(self.out_dir, exist_ok=True)
        loss_str = f"{loss_value:.4f}"
        path = os.path.join(self.out_dir, f"model_step{step}_loss{loss_str}.pt")
        to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        if self.accelerator:
            self.accelerator.wait_for_everyone()
        th.save(to_save.state_dict(), path)
        print(f"[Checkpoint] Saved model to {path}")
