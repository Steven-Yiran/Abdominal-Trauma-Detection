import torch
import pandas as pd
from tqdm import tqdm

from model_zoo.injury_classification import define_model
from datasets.injury_classification_2d_dataset import InjuryClassification2DDataset

def infer_frame_organs(config):
    target_columns = ['patient_id', 'series', 'frame', 'pred_liver', 'pred_spleen', 'pred_kidney', 'pred_bowel']
    df = pd.read_csv(config.segmentations_csv)
    df = df[target_columns]
    return df

def infer_frame_injuries(model, inference_dataset, activations, config):
    """Infer frame-level injuries.
    """
    model.eval()

    results = []
    pbar = tqdm(total=len(inference_dataset), desc='Inference')
    for i in range(len(inference_dataset)):
        images, metadata = inference_dataset[i]['image'], inference_dataset[i]['metadata']
        patient_id = metadata['patient_id']
        series_id = metadata['series_id']
        frame_id = metadata['frame_ids']

        #images = torch.tensor(images).unsqueeze(0)
        images = images.unsqueeze(0)
        images = images.to(config.device)
        outputs = model(images)
        
        # Extract the probabilities from the model outputs
        probabilities = [
            activations[key](value).detach().numpy().flatten()
            for key, value in outputs.items()
        ]

        flattened_probabilities = [item for sublist in probabilities for item in sublist]

        results.append({
            'patient_id': patient_id,
            'series': series_id,
            'frame': frame_id,
            **dict(zip(config.class_names, flattened_probabilities))
        })
        pbar.update(1)

    pbar.close()
    
    return results
