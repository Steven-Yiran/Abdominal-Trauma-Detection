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
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = torch.randint(0, h - new_h, (1,)).item()
        left = torch.randint(0, w - new_w, (1,)).item()

        image = image[top: top + new_h,
                      left: left + new_w]

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