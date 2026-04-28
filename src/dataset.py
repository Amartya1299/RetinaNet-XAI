import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

class DRDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.df.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        label = self.df.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloaders(csv_path, data_dir, batch_size=16):
    df = pd.read_csv(csv_path)
    
    # Stratified Split: 75% Train, 25% (Val + Test) [cite: 75]
    train_df, temp_df = train_test_split(
        df, test_size=0.25, stratify=df['diagnosis'], random_state=42
    )
    # Split the 25% into 15% Val and 10% Test [cite: 75]
    val_df, test_df = train_test_split(
        temp_df, test_size=0.40, stratify=temp_df['diagnosis'], random_state=42
    )

    # Weighted Sampling to fix class imbalance
    class_counts = train_df['diagnosis'].value_counts().to_dict()
    weights = [1.0 / class_counts[label] for label in train_df['diagnosis']]
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), len(weights))

    # Transformations [cite: 76]
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(DRDataset(train_df, data_dir, train_transform), 
                              batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(DRDataset(val_df, data_dir, train_transform), batch_size=batch_size)
    test_loader = DataLoader(DRDataset(test_df, data_dir, train_transform), batch_size=batch_size)

    return train_loader, val_loader, test_loader
