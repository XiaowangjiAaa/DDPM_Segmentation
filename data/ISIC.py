import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

class ISICDataset(Dataset):
    def __init__(self, args, data_path, transform=None, mode='Train'):
        csv_name = f'ISBI2016_ISIC_Part1_{mode}_GroundTruth.csv'
        csv_path = os.path.join(data_path, csv_name)

        # 读取为字符串，防止 int+str 报错
        df = pd.read_csv(os.path.join(data_path, csv_name), dtype=str)
        self.name_list = df["img"].tolist()
        self.label_list = df["seg"].tolist()

        self.data_path = data_path
        self.transform = transform or transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        # 防止 int 类型导致路径拼接失败
        img_rel = str(self.name_list[idx])
        msk_rel = str(self.label_list[idx])

        img_path = os.path.join(self.data_path, img_rel)
        msk_path = os.path.join(self.data_path, msk_rel)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # 同步 transform
        state = torch.get_rng_state()
        img = self.transform(img)
        torch.set_rng_state(state)
        mask = self.transform(mask)

        name = os.path.basename(img_path)

        return {
            "image": img,         # [3, H, W]
            "mask": mask.long(),  # [1, H, W]
            "name": name,
        }
    
