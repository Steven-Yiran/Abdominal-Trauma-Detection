"""
Load and preprocess the data from the data directory
Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
import torch
import pandas as pd
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

        # Dictionary for label index to label name
        self.idx_to_label = {}
        self.label_to_idx = {}
        label_names = self.patient_frame.iloc[:, 1:14].columns.astype('str')
        for i, label in enumerate(label_names):
            self.idx_to_label[i] = label
            self.label_to_idx[label] = i

    def __len__(self):
        return len(self.patient_frame) * len(self.classes)
    
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
        img = read_image(image_path)
        img = img.float() / 255.0
        labels = self.patient_frame.iloc[idx, 1:14].values.astype('float')

        if self.transform:
            img = self.transform(img)
        
        return img, labels

