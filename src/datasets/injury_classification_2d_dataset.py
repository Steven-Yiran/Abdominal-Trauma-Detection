
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
        for i in range(len(raw_dataset)):
            (
                image_paths,
                labels,
                metadata
            ) = raw_dataset[i] 

            #if sample:
                # sample(metadata['patient_id'], metadata['series_id'], metadata['frame_ids'])
                # frame_ids: [0, 1, 4, 5, 6,...]
                # image: [image1, image2, image5, image6, image7,...]
                # frame id 6 -> 4
                #images = sample(images)


            for i in range(len(image_paths)):
                data = {
                    'image_path': image_paths[i],
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
        


if __name__ == '__main__':
    from raw_dataset import RawDataset
    raw_dataset = RawDataset()

    dataset = InjuryClassification2DDataset(raw_dataset, lambda x: x)

    for sample in dataset:
        print(sample)