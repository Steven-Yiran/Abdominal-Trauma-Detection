
import torch
import numpy as np
import sampling

from torch.utils.data import Dataset
from raw_dataset import RawDataset


class InjuryClassification2DDataset(Dataset):
    '''
    Dataset for training the "2D injury classification model".
    '''
    
    def __init__(self, raw_dataset: RawDataset, sample):
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
            #print("sampled data size = ", len(sampled_images))
            for image in sampled_images:
                self.pairs.append({
                    'image': image,
                    'labels': labels
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
        `{'image': np.array, 'labels': {'bowel': np.float32, 'extravasation': np.float32, 'kidney': np.array, 'liver': np.array, 'spleen': np.array}}`
        '''
        return self.pairs[index]
        


if __name__ == '__main__':
    raw_dataset = RawDataset()
    dataset = InjuryClassification2DDataset(raw_dataset, sampling.organ_based_sampleing)
    #dataset = InjuryClassification2DDataset(raw_dataset, sampling.uniformly_random_sampling)

    for i in range(len(dataset)):

        # Print each sampled frame
        print(dataset[i]['image'])
        # Print each label of sampled frame
        print(dataset[i]['labels'])

