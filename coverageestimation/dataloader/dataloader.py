import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import re

class CoverageMapDataset(Dataset):
    def __init__(self, xlsx_file, root_dir, transform=None, colored_transform=None, mode="rss", render_mode="gray_scale"):
        self.data_frame = pd.read_excel(xlsx_file)
        self.root_dir = root_dir
        self.transform = transform
        self.colored_transform = colored_transform
        self.mode = mode
        self.render_mode = render_mode

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_map_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0].lstrip("./"))
        frequency_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1].lstrip("./"))
        power_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2].lstrip("./"))
        transmitter_locations_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 5].lstrip("./"))
        transmitter_height_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 6].lstrip("./"))
        rss_map_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 9].lstrip("./"))
        path_gain_map_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 10].lstrip("./"))
        sinr_map_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 11].lstrip("./"))

        base_map = Image.open(base_map_path)
        frequency = Image.open(frequency_path)
        power = Image.open(power_path)
        transmitter_locations = Image.open(transmitter_locations_path)
        transmitter_height = Image.open(transmitter_height_path)
        rss_map = Image.open(rss_map_path)
        sinr_map = Image.open(sinr_map_path)
        try:
            path_gain_map = Image.open(path_gain_map_path)
        except FileNotFoundError:
            alt_path = path_gain_map_path.replace("path_gain_", "path_gain")
            path_gain_map = Image.open(alt_path)

        if self.transform:
            base_map = self.transform(base_map)
            frequency = self.transform(frequency)
            power = self.transform(power)
            transmitter_locations = self.transform(transmitter_locations)
            transmitter_height = self.transform(transmitter_height)
            if self.render_mode != "gray_scale":
                rss_map = self.colored_transform(rss_map)
                sinr_map = self.colored_transform(sinr_map)
                path_gain_map = self.colored_transform(path_gain_map)
            else:
                rss_map = self.transform(rss_map)
                sinr_map = self.transform(sinr_map)
                path_gain_map = self.transform(path_gain_map)


        input_tensor = torch.cat((base_map, 
                                  frequency, 
                                  power,
                                  transmitter_locations,
                                  transmitter_height), dim=0)
        
        if self.mode == "rss":
            return input_tensor, rss_map
        elif self.mode=="sinr":
            return input_tensor, sinr_map
        elif self.mode=="path_gain":
            return input_tensor, path_gain_map
        
        output_tensor = torch.cat([rss_map,
                                   path_gain_map], dim=0)
        

        return input_tensor, output_tensor


def get_dataloader(xlsx_file, root_dir, batch_size=32, shuffle=True, num_workers=4, render_mode="gray_scale", mode="rss"):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    colored_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
    ])

    dataset = CoverageMapDataset(xlsx_file=xlsx_file, root_dir=root_dir, transform=transform, colored_transform=colored_transform, mode=mode, render_mode=render_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader