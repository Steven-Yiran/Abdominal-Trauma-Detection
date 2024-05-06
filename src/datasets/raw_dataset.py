import pandas as pd
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from torchvision.io import read_image


patient_id_column_name = 'patient_id'
bowel_healthy_column_name = 'bowel_healthy'
extravasation_healthy_column_name = 'extravasation_healthy'

kidney_healthy_column_name = 'kidney_healthy'
kidney_low_column_name = 'kidney_low'
kidney_high_column_name = 'kidney_high'

liver_healthy_column_name = 'liver_healthy'
liver_low_column_name = 'liver_low'
liver_high_column_name = 'liver_high'

spleen_healthy_column_name = 'spleen_healthy'
spleen_low_column_name = 'spleen_low'
spleen_high_column_name = 'spleen_high'

class RawDataset():
    '''
    Raw dataset.
    '''
    
    def __init__(self, csv_path, image_dir):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.table = pd.read_csv(csv_path)

    def __len__(self):
        '''
        Returns the number of patients
        '''
        return len(self.table)

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
        
        row = self.table.iloc[index]

        patient_id = row[patient_id_column_name]
        patient_path = f'{self.image_dir}/train_images/{patient_id}'
        # select the first series of the patient
        series_id = int(os.listdir(patient_path)[0])
        series_path = os.path.join(patient_path, str(series_id))
        frame_ids = sorted([int(file_name.split('.')[0]) for file_name in os.listdir(series_path)])
        image_paths = sorted([os.path.join(series_path, file_name) for file_name in os.listdir(series_path)])

        # images = np.array([read_image(image_path).to(torch.float32) / 255.0 for image_path in image_paths])

        bowel_healthy = row[bowel_healthy_column_name].astype(np.float32)
        
        extravasation_healthy = row[extravasation_healthy_column_name].astype(np.float32)

        kidney_condition = np.array(
            [
                row[kidney_healthy_column_name],
                row[kidney_low_column_name],
                row[kidney_high_column_name]
            ],
            dtype=np.float32
        )

        liver_condition = np.array(
            [
                row[liver_healthy_column_name],
                row[liver_low_column_name],
                row[liver_high_column_name]
            ],
            dtype=np.float32
        )

        spleen_condition = np.array(
            [
                row[spleen_healthy_column_name],
                row[spleen_low_column_name],
                row[spleen_high_column_name]
            ],
            dtype=np.float32
        )

        labels = {
            'bowel': bowel_healthy,
            'extravasation': extravasation_healthy,
            'kidney': kidney_condition,
            'liver': liver_condition,
            'spleen': spleen_condition
        }

        metadata = {
            'patient_id': patient_id,
            'series_id': series_id,
            'frame_ids': frame_ids
        }

        return (
            image_paths,
            labels,
            metadata
        )

if __name__ == '__main__':
    dataset = RawDataset()
    print(len(dataset))
    for i, (images, bowel_healthy, extravasation_healthy, kidney_condition, liver_condition, spleen_condition) in enumerate(dataset):
        print(f'### sample {i} ###')
        print(images.shape)
        print(images.dtype)
        