import pandas as pd
import random

class SamplingStrategies:
    def __init__(self, strategy_name, num_frames, segmentations_csv, threshold=0.5):
        self.strategy_name = strategy_name
        self.num_frames = num_frames
        self.segmentations_csv = segmentations_csv
        self.threshold = threshold

    def __call__(self, *args, **kwds):
        return self.getSamplingStrategy()(*args, **kwds)

    def getSamplingStrategy(self):
        match self.strategy_name:
            case 'Random':
                return self.random_sampling
            case 'OrganBased':
                return self.organ_based_sampling
            case 'Sequential':
                return self.sequential_sampling
            case 'Uniform':
                return self.uniform_sampling
            case 'All':
                return self.all_sampling
            case _:
                return self.uniform_sampling
            
    def random_sampling(self, metadata):
        """
        Uniformly sample frame index for a given patient and series.
        """
        frames = len(metadata['frame_ids'])
        if frames < self.num_frames:
            print(f'Patient {metadata["patient_id"]} has less than {self.num_frames} frames with segmentations.')
            return []
        return random.sample(range(frames), self.num_frames)
    
        
    def uniform_sampling(self, metadata):
        """
        Uniformly sample frame index for a given patient and series.
        """
        frames = len(metadata['frame_ids'])
        if frames < self.num_frames:
            print(f'Patient {metadata["patient_id"]} has less than {self.num_frames} frames with segmentations.')
            return []
        return list(range(self.num_frames))
        
    def sequential_sampling(self, metadata):
        """
        Sequentially sample frame index for a given patient and series.
        """
        frames = len(metadata['frame_ids'])
        if frames < self.num_frames:
            print(f'Patient {metadata["patient_id"]} has less than {self.num_frames} frames with segmentations.')
            return []
        stride = frames // self.num_frames
        return list(range(0, frames, stride))[:self.num_frames]
    
    def all_sampling(self, metadata):
        """
        Sample all frames for a given patient and series.
        """
        return list(range(len(metadata['frame_ids'])))
    
    def organ_based_sampling(self, metadata):
        table = pd.read_csv(self.segmentations_csv)
        patient_id = metadata['patient_id']
        series_id = metadata['series_id']
        frame_ids = metadata['frame_ids']

        # list of relevant frames
        liver = []
        spleen = [] 
        bowel = [] 
        kidney = []
        
        for i, frame_id in enumerate(frame_ids):
            row = table[(table['patient_id'] == patient_id) & (table['series'] == series_id) & (table['frame'] == frame_id)]
            row = row.iloc[0]
            if len(row) == 0:
                continue

            if float(row['pred_liver']) >= self.threshold:
                liver.append(i)
            if float(row['pred_spleen']) >= self.threshold:
                spleen.append(i)
            if float(row['pred_bowel']) >= self.threshold:
                bowel.append(i)
            if float(row['pred_kidney']) >= self.threshold:
                kidney.append(i)

        frame_per_organ = self.num_frames // 4
        liver = random.sample(liver, min(frame_per_organ, len(liver)))
        spleen = random.sample(spleen, min(frame_per_organ, len(spleen)))
        bowel = random.sample(bowel, min(frame_per_organ, len(bowel)))
        kidney = random.sample(kidney, min(frame_per_organ, len(kidney)))
        return liver + spleen + bowel + kidney

    
    # def _get_number_of_frames(self, metadata):
    #     """
    #     Get the number of frames with segmentations for a given patient and series."""
    #     table = pd.read_csv(self.segmentations_csv)
    #     request_patient_id = metadata['patient_id']
    #     request_patient_series_id = metadata['series_id']

    #     filtered_table = table[(table['patient_id'] == request_patient_id) & (table['series'] == request_patient_series_id)]
    #     return len(filtered_table)














    