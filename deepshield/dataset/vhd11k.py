import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.transforms as T


DECISION_MAP = {
    "yes": 1,
    "no": 0
}


def get_transforms(img_size=512):
    transform_augment = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.RandomRotation(degrees=15)],
            p=0.3
        ),
        T.RandomApply(
            [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)],
            p=0.4
        ),
        T.RandomApply(
            [T.RandomAutocontrast()],
            p=0.2
        ),
        T.RandomApply(
            [T.RandomPerspective(distortion_scale=0.5)],
            p=0.2
        ),
        T.ToTensor()
    ])
    
    return transform_augment

class VHD11K(Dataset):
    def __init__(self, data_root, transform=None):
        """
        Args:
            dataset_dir (str): Path to the dataset directory containing images.
            labels_csv (str): Path to the CSV file containing image file names and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        
        self.images_path = os.path.join(self.data_root, "data")
        self.labels_csv = os.path.join(self.data_root, "labels.csv")
        
        
        self.labels_df = pd.read_csv(self.labels_csv)
        self.labels_df['harmIdx'] = pd.factorize(self.labels_df['harmfulType'])[0]
        
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)
    
    def get_num_labels(self):
        return self.labels_df['harmIdx'].nunique()

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert("RGB")
        
        row = self.labels_df.iloc[idx]
        labels = {
            "isHarmful": torch.tensor(DECISION_MAP[row["decision"]], dtype=torch.int64),
            "harmIdx": torch.tensor(row["harmIdx"], dtype=torch.int64),
            "harmDesc": row["harmfulType"]
        }
        
        if self.transform:
            image = self.transform(image)
        
        return {'label': labels["isHarmful"], 'pixel_values': image}
    
    
if __name__ == "__main__":
    my_dataset = VHD11K(
        data_root="/home/bahar/voxel_hackathon/dataset",
        transform=get_transforms()
    )

    print(my_dataset[0])