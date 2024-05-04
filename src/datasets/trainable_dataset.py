import pandas as pd
import sys
import os
import numpy as np
import torch
import random
from raw_dataset import RawDataset
from torch.utils.data import Dataset
from torchvision.io import read_image

NUM_IMAGES_TO_PICK = 30

class StrategiesAppliedDataset(Dataset):

    def __init__(self):
        self.dataset = RawDataset()
        self.table = pd.read_csv('../data/train.csv')

    def __getitem__(self, strategy):
        return self.applyRandomSamplingStrategy(strategy)
            
    def applyRandomSamplingStrategy(self, strategy):
        number_of_pid = len(self.table)
        data = []
        for pid in range(number_of_pid):
            index_list =self.generate_image_list(strategy)
            print("pid = ", pid," strategy = ",strategy, " list = ", index_list)
            image = [self.dataset[pid][0][i] for i in index_list]
            image = np.array(image)
            bowel_healthy = self.dataset[pid][1]
            extravasation_healthy = self.dataset[pid][2]
            kidney_condition = self.dataset[pid][3]
            liver_condition = self.dataset[pid][4]
            spleen_condition = self.dataset[pid][5]
            p_info = (
                image,
                bowel_healthy,
                extravasation_healthy,
                kidney_condition,
                liver_condition,
                spleen_condition,
            )
            data.append(p_info)
        return data
    

    def get_upper_bond(self):
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
        return 0
    
    def generate_image_list(self, strategy):
        upper_bond = self.get_upper_bond()
        match strategy:
            case 'random':
                index_list = sorted(np.random.choice(upper_bond, size=NUM_IMAGES_TO_PICK, replace=False))
                return index_list
            case 'normal':
                selected_indices = set()
                while len(selected_indices) < NUM_IMAGES_TO_PICK:
                    normal_distribution_index = np.random.normal(upper_bond/2 - 1, upper_bond/6)
                    rounded_index = int(round(normal_distribution_index))
                    if 0 <= rounded_index < upper_bond and rounded_index not in selected_indices:
                        selected_indices.add(rounded_index)
                index_list = sorted(selected_indices)
                return index_list
            case _ :
                raise ValueError ('invalid strategy was given')
        

if __name__ == '__main__':
    test = StrategiesAppliedDataset()
    data1 = test["normal"]
    data2 = test["random"]
    for i, (images, bowel_healthy, extravasation_healthy, kidney_condition, liver_condition, spleen_condition) in enumerate(data1):
        print(f'### normal strategy sample {i} ###')
        print(images.shape)
        print(images.dtype)

#     for i, (images, bowel_healthy, extravasation_healthy, kidney_condition, liver_condition, spleen_condition) in enumerate(data2):
#         print(f'### random strategy sample {i} ###')
#         print(images.shape)
#         print(images.dtype)


