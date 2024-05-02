import torch
import torch.nn as nn
from tqdm import tqdm

from model_zoo.injury_classification import define_model
from datasets.raw_dataset import RawDataset
from datasets.injury_classification_2d_dataset import InjuryClassification2DDataset

def infer_frame_injuries(config):
    """Infer frame-level injuries and save the results to a CSV file.
    """
    patient_dataset = RawDataset(is_train=False)
    inference_dataset = InjuryClassification2DDataset(patient_dataset, lambda x: x, is_train=False)
    
    model = define_model(config.model_name)
    model.load_state_dict(torch.load(f'{config.model_checkpoint_name}.pth'))

    model.eval()

    activations = {
        'bowel': nn.Sigmoid(),
        'extravasation': nn.Sigmoid(),
        'kidney': nn.Softmax(dim=1),
        'liver': nn.Softmax(dim=1),
        'spleen': nn.Softmax(dim=1)
    }

    with open(config.results_csv, 'w') as f:
        columns = ['patient_id'] + config.class_names
        f.write(','.join(columns) + '\n')

        for i in tqdm(range(len(inference_dataset))):
            images, patient_id = inference_dataset[i]['image'], inference_dataset[i]['patient_id']

            images = torch.tensor(images).unsqueeze(0)
            outputs = model(images)
            
            # Extract the probabilities from the model outputs
            probabilities = [
                activations[key](value).detach().numpy().flatten()
                for key, value in outputs.items()
            ]

            flattened_probabilities = [item for sublist in probabilities for item in sublist]

            f.write(f'{patient_id},{",".join(map(str, flattened_probabilities))}\n')

    print(f'Inference results saved to {config.results_csv}')