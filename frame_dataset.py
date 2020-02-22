"""
Usage:
Import this dataset by executing:
    from frame_dataset import FrameDataset
Create the dataset by executing:
    train_dataset = FrameDataset("train_sample_frames")
The dataset can be passed to a DataLoader for training/eval:
    train_loader = torch.utils.data.DataLoader(train_dataset,
                     batch_size=16, shuffle=True)

To see images for debugging, create the dataset without a transform:
    train_dataset = FrameDataset("train_sample_frames", transform=None)
Then show an image e.g. with this:
    train_dataset[0][0].show()

Note that the dataset cannot load test directories, since the images
do not have label metadata. These will have to be loaded manually
in case we wish to submit our model's guesses to the Kaggle leaderboards.
"""

import os
import json
import torch
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class FrameDataset(Dataset):
    def __init__(self, image_directory, transform="default"):
        assert(os.path.exists(image_directory))
        self.basedir = image_directory
        with open(os.path.join(image_directory, "metadata.json")) as f_in:
            self.metadata = json.load(f_in)
        self.image_paths = []
        for path in os.listdir(image_directory):
            if path.endswith(".jpg"):
                self.image_paths.append(path)

        self.transform = transform
        if self.transform == "default":
            # This is the transform recommended by PyTorch docs at
            # https://pytorch.org/docs/stable/torchvision/models.html
            self.transform = transforms.Compose([
                transforms.Resize(270), # resize from 1920x1080 to 480x270
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    
    def get_target_from_path(self, path):
        label = 1 if self.metadata[path]["label"] == "FAKE" else 0
        return torch.tensor([label])
    
    def __getitem__(self, index):
        x = Image.open(os.path.join(self.basedir, self.image_paths[index]))
        y = self.get_target_from_path(self.image_paths[index])
        if self.transform:
            x = self.transform(x)
            if x.shape[1] == 270:
                x = x.permute((0, 2, 1))
        
        return x, y
    
    def __len__(self):
        return len(self.image_paths)
