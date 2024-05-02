import os
import sys
import pandas as pd
import numpy as np
from raw_dataset import RawDataset


class Sampling():
    """
        Sampling Class generates np.array(images) after applying strategy
            strategy : str, #strategy name
            images : Input from injury_classification_2d_dataset
            num_frames : (number of frames) to pick up for each pid 
    """

    def __init__(self, strategy, images, num_frames):
        self.strategy = strategy
        self.table = pd.read_csv('../data/train.csv')
        self.num_frames = num_frames
        self.raw_dataset = RawDataset()
        self.raw_images = images

    def __getitem__ (self):
        return self.get_sampled_images()

    
    def __len__(self):
        return len(self.get_sampled_images())

    def get_upper_bond(self):
        """
            This Method returns the Upper Bonds of images selection from `/data/pid/sid...`
        """
        bond = sys.maxsize
        number_of_pid = len(self.table)
        for i in range (number_of_pid):
            row = self.table.iloc[i]
            patient_id = row['patient_id']
            patient_path = f'../data/train_images/{patient_id}'
            series_path = os.path.join(patient_path, os.listdir(patient_path)[0])
            image_paths = sorted([os.path.join(series_path, file_name) for file_name in os.listdir(series_path)])
            bond = min(bond, len(image_paths))              
        if bond != sys.maxsize:
            return bond
        raise ValueError ("Get Upper Bonds of frames failed")
    
    def generate_sampled_images_index_list(self):
        upper_bound = self.get_upper_bond()
        match self.strategy:
            case 'random':
                index_list = sorted(np.random.choice(upper_bound, size = self.num_frames, replace=False))
                return index_list
            case 'normal':
                selected_indices = set()
                while len(selected_indices) < 30:
                    random_number = int(np.random.normal(0, 1) * upper_bound)
                    random_number = np.clip(random_number, 0, upper_bound - 1)
                    if random_number not in selected_indices:
                        selected_indices.add(random_number)
                    
                index_list = sorted(selected_indices)
                return index_list
            case _ :
                raise ValueError ('invalid strategy was given')
    
    def get_sampled_images(self):
        index = self.generate_sampled_images_index_list()
        images = self.raw_images
        return np.array([images[i] for i in index])