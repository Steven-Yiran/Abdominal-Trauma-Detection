
import torch
import numpy as np

from torch.utils.data import Dataset
from datasets.raw_dataset import RawDataset
from torchvision.io import read_image


class InjuryClassification2DDataset(Dataset):
    '''
    Dataset for training the "2D injury classification model".
    '''
    
    def __init__(self, raw_dataset: RawDataset, sample=None, transform=None, is_train=True):
        self.transform = transform
        self.is_train = is_train
        self.pairs = []
        for pid in range(len(raw_dataset)):
            (
                image_paths,
                labels,
                metadata
            ) = raw_dataset[pid]

            if sample:
                index = sample(metadata)
            else:
                index = range(len(image_paths))

            for fid in index:
                data = {
                    'image_path': image_paths[fid],
                    'label': labels
                }
                if not is_train:
                    data['metadata'] = {
                        'patient_id': metadata['patient_id'],
                        'series_id': metadata['series_id'],
                        'frame_ids': metadata['frame_ids'][fid],
                    }
                self.pairs.append(data)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        '''
        Get the `index`-th image-label pair.

        Parameters
        ----------
        `index`: the index

        Returns
        -------
        `{'image': np.array, 'label': {'bowel': np.float32, 'extravasation': np.float32, 'kidney': np.array, 'liver': np.array, 'spleen': np.array}}`
        '''
        image_path = self.pairs[index]['image_path']
        label = self.pairs[index]['label']

        image = read_image(image_path).to(torch.float32) / 255.0
        data = {
            'image': image,
            'label': label
        }
        if not self.is_train:
            data['metadata'] = self.pairs[index]['metadata']

        if self.transform:
            return self.transform(data)

        return data