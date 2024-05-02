import os
import sys
import pandas as pd
import numpy as np
from raw_dataset import RawDataset

def normal_distribution_sampling(images):
    upper_bound = get_upper_bond()
    index_list = sorted(np.random.choice(upper_bound, size = int(upper_bound*0.7), replace=False))
    return np.array([images[i] for i in index_list])



def uniformly_random_sampling(images):
    upper_bound = get_upper_bond()
    selected_indices = set()
    while len(selected_indices) < int(upper_bound * 0.7):
        random_number = int(np.random.normal(0, 1) * upper_bound)
        random_number = np.clip(random_number, 0, upper_bound - 1)
        if random_number not in selected_indices:
            selected_indices.add(random_number)
    index_list = sorted(selected_indices)
    return np.array([images[i] for i in index_list])


def get_upper_bond():
    """
        This Method returns the Upper Bonds of images selection from `/data/pid/sid...`
    """
    table = pd.read_csv('../data/train.csv')
    bond = sys.maxsize
    number_of_pid = len(table)
    for i in range (number_of_pid):
        row = table.iloc[i]
        patient_id = row['patient_id']
        patient_path = f'../data/train_images/{patient_id}'
        series_path = os.path.join(patient_path, os.listdir(patient_path)[0])
        image_paths = sorted([os.path.join(series_path, file_name) for file_name in os.listdir(series_path)])
        bond = min(bond, len(image_paths))              
    if bond != sys.maxsize:
        return bond
    raise ValueError ("Get Upper Bonds of frames failed")