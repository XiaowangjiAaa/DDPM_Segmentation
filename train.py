import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torchvision import transforms

from models.unet_model import UNetModel_newpreview
from diffusion.gaussian_diffusion import GaussianDiffusion
from diffusion.schedule import get_named_beta_schedule
from train_util.seg_train_loop import TrainLoop
from utils import logger
from data.ISIC import ISICDataset

def main():
    args = create_argparser().parse_args()
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    logger.configure(
        log_dir=args.out_dir,
        project=args.wandb_project,
        use_wandb=args.use_wandb,
        config=vars(args)
    )
    accelerator = Accelerator()
    logger.log("Loading ISIC dataset...")

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ISICDataset(args, args.data_dir, transform, mode='Train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = ISICDataset(args, args.val_data_dir, transform, mode='Val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    logger.log("Creating model and diffusion...")
    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    diffusion = GaussianDiffusion(betas)
    model = UNetModel_newpreview(
        image_size=args.image_size,
        in_channels=args.in_ch,
        model_channels=args.model_channels,
        out_channels=args.out_channels,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=[int(res) for res in args.attention_resolutions.split(',')],
        dropout=args.dropout,
        channel_mult=tuple([int(m) for m in args.channel_mult.split(',')]),
        use_fp16=args.use_fp16,
        use_checkpoint=False,
        resblock_updown=False,
        use_scale_shift_norm=True,
        num_heads=args.num_heads,
    ).to(accelerator.device)

    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        dataloader=dataloader,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        logger=logger,
        out_dir=args.out_dir,
        val_loader=val_loader,
        val_interval=args.val_interval,
        use_scheduler=args.use_scheduler,
        lr_scheduler_tmax=args.lr_scheduler_tmax,
        accelerator=accelerator,
    ).run_loop()


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser

if __name__ == "__main__":
    main()