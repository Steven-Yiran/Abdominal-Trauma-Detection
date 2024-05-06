"""
Preprocessing functions for the data
"""

import torch
import numpy as np
from torchvision import transforms as T

class ExpandDims(object):
    """Expand the image in a sample to have a third dimension of 1.
    """
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image[:, :, None]
        return {'image': image, 'label': label}


class Resize(object):
    """Resize the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        self.t = T.Resize(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        image = self.t(image)
        return {'image': image, 'label': label}
    
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        self.t = T.RandomCrop(output_size)

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = self.t(image)

        return {'image': image, 'label': label}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
    

class ToTensorDict(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for k, v in label.items():
            if type(v) == np.ndarray:
                label[k] = torch.from_numpy(v)

        return {'image': T.ToTensor()(image),
                'label': label}
    

class ToPILImage(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = image.transpose((1, 2, 0))
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = T.ToPILImage()(image)
        return {'image': image,
                'label': label}