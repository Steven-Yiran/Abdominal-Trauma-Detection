
import torch
import numpy as np

from torch.utils.data import Dataset
from datasets.raw_dataset import RawDataset


class InjuryClassification2DDataset(Dataset):
    '''
    Dataset for training the "2D injury classification model".
    '''
    
    def __init__(self, raw_dataset: RawDataset, sample, transform=None):
        self.transform = transform
        self.pairs = []
        for i in range(len(raw_dataset)):
            (
                images,
                bowel_healthy,
                extravasation_healthy,
                kidney_condition,
                liver_condition,
                spleen_condition
            ) = raw_dataset[i]

            labels = {
                'bowel': bowel_healthy,
                'extravasation': extravasation_healthy,
                'kidney': kidney_condition,
                'liver': liver_condition,
                'spleen': spleen_condition
            }

            sampled_images = sample(images)
            for image in sampled_images:
                self.pairs.append({
                    'image': image,
                    'label': labels
                })

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

    dataset = InjuryClassification2DDataset(raw_dataset, lambda x: x)

    for sample in dataset:
        print(sample)