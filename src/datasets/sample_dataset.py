"""
Load and preprocess the data from the data directory
Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
import torch
import pandas as pd
from skimage import io
from torchvision.io import read_image
from torch.utils.data import Dataset

from params import IMG_TARGETS_EXTENDED, PATIENT_TARGETS

class SamplePatientDataset(Dataset):
    """Sample patient dataset"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patient_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = IMG_TARGETS_EXTENDED

    def __len__(self):
        return len(self.patient_frame)
    
    def __getitem__(self, idx):
        """
        Access item by index.

        Args:
            idx (int): Index of the item to access.

        Returns:
            img (Tensor): Image tensor.
            label (Tensor): Label tensor.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_path = os.path.join(self.img_dir, self.patient_frame.iloc[idx].image_path)
        img = io.imread(image_path)
        label = self.patient_frame.iloc[idx, 1:14].values.astype('float')
        sample = {'image': img, 'label': label}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

