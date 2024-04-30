
import torch
import numpy as np

from torch.utils.data import Dataset
from raw_dataset import RawDataset


class InjuryClassification2DDataset(Dataset):
    '''
    Dataset for training the "2D injury classification model.
    '''
    
    def __init__(self, raw_dataset: RawDataset, sample):
        pairs = []
        for i in range(len(raw_dataset)):
            (
                images,
                bowel_healthy,
                extravasation_healthy,
                kidney_condition,
                liver_condition,
                spleen_condition
            ) = raw_dataset[i]

            sampled_images = sample(images)
            for image in sampled_images:
                label = np.concatenate(
                    [bowel_healthy, extravasation_healthy],
                    kidney_condition,
                    liver_condition,
                    spleen_condition
                )
                
                pairs.append((image, label))
            
        self.pairs = np.array(pairs)

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
        a tuple of `(image, label)`. `label` (shape `(11,)`) is the concatenation of the condition of 5 organs.
        '''
        return self.pairs[index]
        


if __name__ == '__main__':
    raw_dataset = RawDataset()

    dataset = InjuryClassification2DDataset(raw_dataset)