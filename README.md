# DDPM Segmentation

This project implements crack segmentation using a diffusion-based model following the MedSegDiff approach. The diffusion model predicts noise conditioned on the RGB image, while the ground-truth mask is only used as the diffusion target and not concatenated with the input.

Training leverages **Hugging Face Accelerate** for multi-GPU support and uses **Weights & Biases** for logging metrics and visualizing validation predictions.

## Usage

Prepare the ISIC dataset following the directory structure used in `ISICDataset` and edit `configs/default.yaml` for paths and training settings.

Run training with:

```bash
accelerate launch train.py --config configs/default.yaml
```

During training, validation images and predicted masks are logged to W&B if `use_wandb: true` in the config.

See `command.txt` for additional commands.
