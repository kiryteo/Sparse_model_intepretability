import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from glob import glob
from PIL import Image

class ShapesDataset(Dataset):

    def __init__(self, data_dir: str, transforms=None):
        shapes = sorted(os.listdir(data_dir))
        self.transforms = transforms
        self.data_paths = []
        self.labels = []
        for i, shape in enumerate(shapes):
            data_paths = glob(os.path.join(data_dir, shape, "*"))
            self.data_paths += data_paths
            self.labels += [i for _ in range(len(data_paths))]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        data_path, label = self.data_paths[index], self.labels[index]
        image = Image.open(data_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

dataset = ShapesDataset("/localhome/asa420/Lottery-Ticket-Hypothesis-in-Pytorch/data/shapes/")
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2], generator=torch.Generator().manual_seed(42))
train_data_loader = DataLoader(train_dataset, 16, shuffle=True)
val_data_loader = DataLoader(val_dataset, 16, shuffle=False)
test_data_loader = DataLoader(test_dataset, 16, shuffle=False)