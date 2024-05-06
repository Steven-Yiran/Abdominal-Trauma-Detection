import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from datasets.series_injury_classification_dataset import SeriesInjuryClassificationDataset
from model_zoo.series_classification import define_model


def train(config):
    generator = torch.Generator().manual_seed(42)

    # Load the dataset
    train_dataset = SeriesInjuryClassificationDataset(
        frame_predictions_csv=config.frame_label_path,
        patient_labels_csv=config.train_csv,
        max_series_length=config.max_series_length
    )
    print(f'Loaded {len(train_dataset)} training samples')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator
    )

    # Define the model
    model = define_model(config)
    model.to(config.device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Define the loss function
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(config.num_epochs):
        pbar = tqdm(total=len(train_dataloader), desc=f'Training Epoch {epoch}')
        for i, (features, labels) in enumerate(train_dataloader):
            features = features.to(config.device)
            labels = labels.to(config.device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.update(1)
        pbar.close()
        print(f'Epoch {epoch} Loss: {loss.item()}')