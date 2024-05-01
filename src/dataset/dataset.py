"""
Load and preprocess the data from the data directory
Reference: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""
import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


patient_id_column_name = 'patient_id'
target_columns = {
    'bowel': 'bowel_healthy',
    'extravasation': 'extravasation_healthy',
    'kidney': ['kidney_healthy', 'kidney_low', 'kidney_high'],
    'liver': ['liver_healthy', 'liver_low', 'liver_high'],
    'spleen': ['spleen_healthy', 'spleen_low', 'spleen_high']
}

# class WrapperClass

# class PatientDatasetBaseline

# class PatientDatasetRandom

# class PatientDatasetRuleBased
# p(organs)

# class PatientDatasetAttention

class PatientDataset(Dataset):
    """
    KSNA Abdominal Injury Detection Dataset
    """

    def __init__(self, csv_file, img_dir, num_classes=11, transform=None):
        self.table = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.num_classes = num_classes
        self.transform = transform

        # TODO: loading wil be different for training and testing

    def __len__(self):
        return len(self.table) * self.num_classes # 2 * 5 = 10

    def __getitem__(self, index):
        '''
        Get the `index`-th patient's data.

        Parameters
        ----------
        - `index`: patient's index as in `train.csv` (not `patient_id`)

        Returns
        -------
        A tuple consisting of:
        - `images`: the images of the first series of this patient (for simplicity), of shape `(number of images, number of channels, height, width)`
        - `bowel_healthy`: a float, either 0 or 1, representing if bowel is healthy
        - `extravasation_healthy`: a float, either 0 or 1, representing if extravasation is healthy
        - `kidney_condition`: a one-hot NumPy array of length 3, representing kidney_healthy, kidney_low, kidney_high
        - `liver_condition`: a one-hot NumPy array of length 3, representing liver_healthy, liver_low, liver_high
        - `spleen_condition`: a one-hot NumPy array of length 3, representing spleen_healthy, spleen_low, spleen_high
        '''
        target_idx = index % self.num_classes
        patient_idx = index // self.num_classes

        row = self.table.iloc[patient_idx]

        patient_id = row[patient_id_column_name]
        patient_path = os.path.join(self.img_dir, 'train_images', str(patient_id))
        series_path = os.path.join(patient_path, os.listdir(patient_path)[0])

        # TODO: Implement sampling strategy here
        # image_paths = [os.path.join(series_path, file_name) for file_name in os.listdir(series_path)]
        # images = np.array([read_image(image_path).to(torch.float32) / 255.0 for image_path in image_paths])

        # get the first image of the series
        image_path = os.path.join(series_path, os.listdir(series_path)[0])
        image = io.imread(image_path)

        bowel_healthy = row[target_columns['bowel']].astype(np.float32)
        
        extravasation_healthy = row[target_columns['extravasation']].astype(np.float32)

        kidney_condition = np.array(
            [row[target_columns['kidney']]],
            dtype=np.float32
        )

        liver_condition = np.array(
            [row[target_columns['liver']]],
            dtype=np.float32
        )

        spleen_condition = np.array(
            [row[target_columns['spleen']]],
            dtype=np.float32
        )

        labels = {
            'bowel': bowel_healthy,
            'extravasation': extravasation_healthy,
            'kidney': kidney_condition,
            'liver': liver_condition,
            'spleen': spleen_condition
        }

        sample = {
            'image': image,
            'label': labels
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
    


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
    
if __name__ == '__main__':
    dataset = PatientDataset("../data/train.csv", "../data")
    for i, data in enumerate(dataset):
        print(f'### sample {i} ###')
        print(data['images'].shape)
        print(data['images'].dtype)   