from torch.utils.data import Dataset, DataLoader, random_split
import os
from glob import glob
from PIL import Image
import torch
class ShapesDataset(Dataset):
    def __init__(self, data_dir: str, transforms):
        shapes = sorted(os.listdir(data_dir))
        self.transforms = transforms
        self.data_paths = []
        self.labels = []
        for i, shape in enumerate(shapes):
            data_paths = glob(os.path.join(data_dir, shape, "*"))
            self.data_paths += data_paths
            self.labels += [i for _ in range(len(data_paths))]
    def __getitem__(self, index):
        data_path, label = self.data_paths[index], self.labels[index]
        image = Image.open(data_path)
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label
    def __len__(self):
        return len(self.data_paths)

def get_dataloaders(data_path, transforms, batch_size):
    dataset = ShapesDataset(data_path, transforms)
    train_dataset_num = int(len(dataset)*0.6)
    val_dataset_num = int(len(dataset)*0.2)
    test_dataset_num = len(dataset) - val_dataset_num - train_dataset_num
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_dataset_num, val_dataset_num, test_dataset_num], generator=torch.Generator().manual_seed(42))
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_data_loader, val_data_loader, test_data_loader
