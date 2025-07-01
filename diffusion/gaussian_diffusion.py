# diffusion/gaussian_diffusion.py
# 包含 DDPM 的核心逻辑：前向加噪、反向去噪采样、loss 构造

import torch as th
import torch.nn.functional as F
import numpy as np

class GaussianDiffusion:
    def __init__(self, betas, model_mean_type="eps", model_var_type="fixedlarge"):
        self.betas = betas
        self.num_timesteps = len(betas)
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_recip_alphas = np.sqrt(1.0 / alphas)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = th.randn_like(x_start)
        sqrt_alphas_cumprod_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t, noise=None, cond=None):
        if noise is None:
            noise = th.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if cond is not None:
            model_in = th.cat([cond, x_noisy], dim=1)
        else:
            model_in = x_noisy
        predicted = model(model_in, t)[0]  # 忽略 cal 输出
        return F.mse_loss(predicted, noise)

    def p_sample(self, model, x, t, cond=None):
        if cond is not None:
            model_in = th.cat([cond, x], dim=1)
        else:
            model_in = x
        eps = model(model_in, t)[0]
        coef1 = _extract_into_tensor(self.sqrt_recip_alphas, t, x.shape)
        coef2 = _extract_into_tensor(self.betas, t, x.shape) / _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        mean = coef1 * (x - coef2 * eps)
        mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        noise = th.randn_like(x)
        var = _extract_into_tensor(self.posterior_variance, t, x.shape)
        sample = mean + mask * th.sqrt(var) * noise
        return sample

    def p_sample_loop(self, model, shape, cond):
        device = cond.device
        img = th.randn(shape, device=device)
        for i in reversed(range(self.num_timesteps)):
            t = th.tensor([i] * shape[0], device=device, dtype=th.long)
            img = self.p_sample(model, img, t, cond=cond)
        return img


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = th.from_numpy(arr).to(device=timesteps.device, dtype=th.float32)[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)