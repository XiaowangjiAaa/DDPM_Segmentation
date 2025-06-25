import builtins
import datetime
import os
import wandb
from PIL import Image

class Logger:
    def __init__(self, log_dir=None, project=None, use_wandb=False, config=None):
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_file = open(os.path.join(log_dir, f"log_{time_stamp}.txt"), 'a')
            self.image_dir = os.path.join(log_dir, "samples")
            os.makedirs(self.image_dir, exist_ok=True)
        else:
            self.log_file = None
            self.image_dir = "./samples"

        if use_wandb:
            wandb.init(project=project or "ddpm_segmentation", config=config)

    def log(self, message, stdout=True):
        time_prefix = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        full_msg = f"{time_prefix} {message}"
        if stdout:
            print(full_msg)
        if self.log_file:
            self.log_file.write(full_msg + "\n")
            self.log_file.flush()

    def log_metrics(self, metrics_dict, step=None):
        if self.use_wandb:
            wandb.log(metrics_dict, step=step)

    def log_images(self, images, names, step=None):
        """
        images: list of np.array or PIL.Image
        names: list of image names (for saving)
        """
        saved_paths = []
        for i, img in enumerate(images):
            name = names[i]
            save_path = os.path.join(self.image_dir, f"{step}_{name}.png")
            if isinstance(img, Image.Image):
                img.save(save_path)
            else:
                Image.fromarray(img).save(save_path)
            saved_paths.append(save_path)

        if self.use_wandb:
            wandb_images = [wandb.Image(Image.open(p), caption=names[i]) for i, p in enumerate(saved_paths)]
            wandb.log({f"val/images": wandb_images}, step=step)

    def close(self):
        if self.log_file:
            self.log_file.close()
        if self.use_wandb:
            wandb.finish()

# 全局 logger 实例
GLOBAL_LOGGER = None

def configure(log_dir=None, project=None, use_wandb=False, config=None):
    global GLOBAL_LOGGER
    GLOBAL_LOGGER = Logger(log_dir, project, use_wandb, config)

def log(msg):
    if GLOBAL_LOGGER is not None:
        GLOBAL_LOGGER.log(msg)
    else:
        print(msg)

def log_metrics(metrics, step=None):
    if GLOBAL_LOGGER is not None:
        GLOBAL_LOGGER.log_metrics(metrics, step)

def log_images(images, names, step=None):
    if GLOBAL_LOGGER is not None:
        GLOBAL_LOGGER.log_images(images, names, step)

def close():
    if GLOBAL_LOGGER is not None:
        GLOBAL_LOGGER.close()
