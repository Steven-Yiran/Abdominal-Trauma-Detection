import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_zoo.models import define_model
from dataset.dataset import SamplePatientDataset
from dataset.transform import Rescale, RandomCrop, ToTensor

def batch_step(
    model,
    criterion,
    optimizer,
    inputs,
    labels,
):
    """
    Perform a single batch step.
    """
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def train(
    config,
):
    """
    Train a injury classification model.
    """
    transform_fn = transforms.Compose([
        Rescale(config.input_size + 
                config.input_size // 8),
        RandomCrop(config.input_size),
        ToTensor()
    ])
    train_dataset = SamplePatientDataset(
        csv_file=config.train_csv,
        img_dir=config.img_dir,
        transform=transform_fn
    )
    # selet the first 100 samples for training
    train_dataset = torch.utils.data.Subset(train_dataset, range(100))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )

    # Define the model
    model = define_model(
        model_name=config.model_name,
        num_classes=config.num_classes
    )

    model.zero_grad()
    model.train()
    model.double()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config.num_epochs):
        running_loss = 0.0
        for data in tqdm(train_dataloader):
            inputs, labels = data['image'], data['labels']
            loss = batch_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                inputs=inputs,
                labels=labels
            )
            running_loss += loss.item()
        print(f"Epoch {epoch} loss: {running_loss / len(train_dataloader)}")