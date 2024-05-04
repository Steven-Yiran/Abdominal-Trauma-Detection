from torch.utils.data import Dataset
from raw_dataset import RawDataset


class SeriesInjuryClassificationDataset(Dataset):
    def __init__(self):
        # load (p_organs, p_injuries, label)
        self.series = []

    def __len__(self):
        return len(self.series)
    
    def __getitem__(self, index):
        return self.series[index]