import os
from torch.utils.data import Dataset


class CorrelationDataset(Dataset):
    def __init__(self, df, image_root, transform=None, target_transform=None):
        self.df = df
        self.image_root = image_root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_root, f"{self.df.iloc[idx, 0]}.png")
        image = read_image(image_path)

        label = self.df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, torch.Tensor([label])
