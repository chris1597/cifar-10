from pydantic import BaseModel, DirectoryPath, FilePath
from typing import Optional, Callable
import pandas as pd
import os
from skimage import io
from torch.utils.data import Dataset

class DatasetHelper(BaseModel):
    csv_file: FilePath
    img_dir: DirectoryPath
    transform: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True

    def create_dataset(self) -> Dataset:
        """Method to create and return the custom image dataset."""
        class CustomImageDataset(Dataset):
            def __init__(self, csv_file, img_dir, transform=None):
                self.img_labels = pd.read_csv(csv_file)
                self.img_dir = img_dir
                self.transform = transform
                # Create a mapping from labels to integers
                self.label_to_int = {label: index for index, label in enumerate(self.img_labels['label'].unique())}
                print(self.label_to_int)

            def __len__(self):
                return len(self.img_labels)

            def __getitem__(self, idx):
                img_path = os.path.join(self.img_dir, str(self.img_labels.iloc[idx, 0]) + '.png')
                image = io.imread(img_path)
                 # Get the string label
                label_str = self.img_labels.iloc[idx, 1] 
                # Convert to integer
                label = self.label_to_int[label_str]
                if self.transform:
                    image = self.transform(image)
                return image, label

        return CustomImageDataset(self.csv_file, self.img_dir, self.transform)