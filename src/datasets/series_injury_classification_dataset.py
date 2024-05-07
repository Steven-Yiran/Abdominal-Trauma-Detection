import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

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


class SeriesInjuryClassificationDataset(Dataset):
    def __init__(self, frame_predictions_csv, patient_labels_csv, max_series_length):
        self.frame_predictions = pd.read_csv(frame_predictions_csv)
        self.patients = self.frame_predictions['patient_id'].unique()
        self.patient_labels = pd.read_csv(patient_labels_csv)
        self.max_series_length = max_series_length

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        patient_id = self.patients[index]
        patient_series_data = self.frame_predictions[self.frame_predictions['patient_id'] == patient_id]
        patient_labels = self.patient_labels[self.patient_labels['patient_id'] == patient_id]

        features = patient_series_data.drop(columns=['patient_id'])
        labels = patient_labels.drop(columns=['patient_id', 'any_injury', 'bowel_injury', 'extravasation_injury'])
        # reduce to one row
        labels = labels.iloc[0]

        # crop features to fix size of 10]
        if len(features) > self.max_series_length:
            features = features[:self.max_series_length]
        elif len(features) < self.max_series_length:
            num_padding_rows = self.max_series_length - len(features)
            padding = np.zeros((num_padding_rows, features.shape[1]))

            padding_df = pd.DataFrame(padding, columns=features.columns)
            features = pd.concat([features, padding_df], ignore_index=True)
    
        return torch.tensor(features.values, dtype=torch.float32), torch.tensor(labels.values, dtype=torch.float32)

