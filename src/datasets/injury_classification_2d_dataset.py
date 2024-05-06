
import torch
import numpy as np

from torch.utils.data import Dataset
from raw_dataset import RawDataset
import sampling

class sampling_config():
        metadata = None
        organSegPath = "../data/p_organ_images/organ_seg/df_images_train_with_seg.csv" #Organ seg path 
        threshold = 0.95 # True threshold
        numFrames = 80 # total number of frames to choice
        strategy = "OrganBased" #"Uniform"  # samping switch
        @classmethod
        def set_data(cls, data):
           cls.metadata = data

class InjuryClassification2DDataset(Dataset):
    '''
    Dataset for training the "2D injury classification model".
    '''
    
    def __init__(self, raw_dataset: RawDataset, sample=None, transform=None, is_train=True):
        self.transform = transform
        self.pairs = []
        for i in range(len(raw_dataset)):
            (
                images,
                labels,
                metadata
            ) = raw_dataset[i] 

            if sample:
                sampling_config.set_data(metadata)
                sample_index = sample(sampling_config)

            for i in sample_index:
                data = {
                    'image': images[i],
                    'label': labels
                }
                if not is_train:
                    data['metadata'] = {
                        'patient_id': metadata['patient_id'],
                        'series_id': metadata['series_id'],
                        'frame_ids': metadata['frame_ids'][i],
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
        if self.transform:
            return self.transform(self.pairs[index])

        return self.pairs[index]
        


if __name__ == '__main__':
    from raw_dataset import RawDataset
    raw_dataset = RawDataset()

    dataset = InjuryClassification2DDataset(raw_dataset, sample = sampling.collectSamplingData)

    for sample in dataset:
        print("sample = ", sample)
