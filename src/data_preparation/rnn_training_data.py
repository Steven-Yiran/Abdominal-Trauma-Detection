import torch
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import os

from datasets.raw_dataset import RawDataset
from datasets.injury_classification_2d_dataset import InjuryClassification2DDataset
from inference.frame_inference import infer_frame_organs, infer_frame_injuries_batch
from model_zoo.injury_classification import define_model
from utils.transform import ToTensor, Resize, RandomCrop, ToPILImage
from datasets.sampling import SamplingStrategies


def generate(config, inference_dataset=None):
    """
    Generate frame-level injury predictions and organ segmentations.
    """
    if not os.path.exists(config.segmentations_csv):
        raise FileNotFoundError(f'File not found: {config.segmentations_csv}')
    
    test_transform = transforms.Compose([
        ToPILImage(),
        Resize(config.input_size),
        RandomCrop(config.input_size),
        ToTensor()
    ])
    if not inference_dataset:
        patient_dataset = RawDataset(csv_path=config.train_csv, image_dir=config.img_dir)
        sample = SamplingStrategies(
            "All",
            config.max_frame_per_patient,
            config.segmentations_csv,
            config.threshold
        )
        inference_dataset = InjuryClassification2DDataset(patient_dataset, sample=sample, is_train=False, transform=test_transform)
    
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    model = define_model(config.model_name)
    model.load_state_dict(torch.load(f'{config.model_checkpoint_name}.pth'))
    model.to(config.device)

    activations = {
        'bowel': nn.Sigmoid(),
        'extravasation': nn.Sigmoid(),
        'kidney': nn.Softmax(dim=1),
        'liver': nn.Softmax(dim=1),
        'spleen': nn.Softmax(dim=1)
    }

    #injury_results = infer_frame_injuries(model, inference_dataset, activations, config)
    injury_results = infer_frame_injuries_batch(model, inference_dataloader, activations, config)
    injury_results_df = pd.DataFrame(injury_results)
    organ_results_df = infer_frame_organs(config)
    
    # inner join injury_results_df and organ_results_df on patient_id, series_id, frame
    injury_results_df['series'] = injury_results_df['series'].astype(int)
    frame_predictions = pd.merge(injury_results_df, organ_results_df, on=['patient_id', 'series', 'frame'], how='inner')

    # drop unnecessary columns
    frame_predictions.drop(columns=['series', 'frame'], inplace=True)

    # save frame_predictions to disk
    frame_predictions.to_csv(config.frame_label_path, index=False)

    print(f'Results saved to {config.frame_label_path}')
