import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

class CoverageMapDataset(Dataset):
    def __init__(self, xlsx_file, root_dir, transform=None):
        self.data_frame = pd.read_excel(xlsx_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        base_map_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        frequency_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])
        #terrain_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 2])
        azimuth_angles_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 3])
        transmitter_tilt_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 4])
        transmitter_locations_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 5])
        transmitter_height_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 6])
        antenna_pattern_vertical_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 7])
        coverage_map_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 9])

        base_map = Image.open(base_map_path)
        frequency = Image.open(frequency_path)
        #terrain = Image.open(terrain_path)
        azimuth_angles = Image.open(azimuth_angles_path)
        transmitter_tilt = Image.open(transmitter_tilt_path)
        transmitter_locations = Image.open(transmitter_locations_path)
        transmitter_height = Image.open(transmitter_height_path)
        antenna_pattern_vertical = Image.open(antenna_pattern_vertical_path)
        coverage_map = Image.open(coverage_map_path)

        if self.transform:
            base_map = self.transform(base_map)
            frequency = self.transform(frequency)
            #terrain = self.transform(terrain)
            azimuth_angles = self.transform(azimuth_angles)
            transmitter_tilt = self.transform(transmitter_tilt)
            transmitter_locations = self.transform(transmitter_locations)
            transmitter_height = self.transform(transmitter_height)
            antenna_pattern_vertical = self.transform(antenna_pattern_vertical)
            coverage_map = self.transform(coverage_map)

        input_tensor = torch.cat((base_map, frequency, azimuth_angles, 
                                  transmitter_tilt, transmitter_locations, transmitter_height, antenna_pattern_vertical), dim=0)

        return input_tensor, coverage_map

def get_dataloader(xlsx_file, root_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset = CoverageMapDataset(xlsx_file=xlsx_file, root_dir=root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader