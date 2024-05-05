import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets.series_injury_classification_dataset import SeriesInjuryClassificationDataset
from model_zoo.series_classification import define_model


def train(config):
    generator = torch.Generator().manual_seed(42)

    # Load the dataset
    train_dataset = SeriesInjuryClassificationDataset(
        frame_predictions_csv=config.frame_label_path,
        patient_labels_csv=config.train_csv
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator
    )

    # Define the model
    model = define_model(config)
    # model.to(config.device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Define the loss function
    criterion = nn.MSELoss()

    # Train the model
    for epoch in range(config.num_epochs):
        for i, (features, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')