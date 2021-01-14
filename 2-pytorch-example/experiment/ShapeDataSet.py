import os
import numpy as np
from matplotlib.pyplot import imread
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float64
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


def _rgb_to_gray(rgb):
    # Convert rgb images to gray images
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class ShapeDataset(Dataset):

    def __init__(self, data_root):
        self.samples = []
        self.to_tensor = transforms.ToTensor()

        directories = os.listdir(data_root)

        for d in directories:
            for k in os.listdir(data_root + d):
                img = imread(data_root + d + "/" + k)
                img = np.asarray(_rgb_to_gray(img), dtype=np.float32)
                self.samples.append((d, img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # return img, target
        data = self.samples[idx]
        label = 0
        if data[0] == 'circles':
            label = 1
        elif data[0] == 'squares':
            label = 2
        elif data[0] == 'triangles':
            label = 3

        # print(data[1])
        # print(label)

        return self.to_tensor(data[1]), tensor(label)  # tensor(label, dtype=float64)


if __name__ == '__main__':
    dataset = ShapeDataset('data/shapes/')
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)

    for i, batch in enumerate(dataloader):
        print(i, batch)
