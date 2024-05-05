import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from model_zoo.injury_classification import define_model
from datasets.raw_dataset import RawDataset
from datasets.injury_classification_2d_dataset import InjuryClassification2DDataset
from utils.transform import ToTensorDict

def compute_loss(outputs, labels, criterion_dict):
    loss = 0.
    for key in outputs.keys():
        loss += criterion_dict[key](outputs[key].squeeze(), labels[key].squeeze())
    return loss


def train_one_epoch(
    model,
    train_dataloader,
    criterion_dict,
    optimizer,
    report_interval=10,
):
    model.zero_grad()
    model.train()

    running_loss = 0.
    last_loss = 0.

    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    for i, data in enumerate(train_dataloader):
        image, labels = data['image'], data['label']
        optimizer.zero_grad()
        outputs = model(image)
        loss = compute_loss(outputs, labels, criterion_dict)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % report_interval == 0:
            last_loss = running_loss / report_interval
            print(f'[{i}/{len(train_dataloader)}]\tLoss: {loss.item():.6f}')
            running_loss = 0.
        
        pbar.update(1)

    return last_loss


def evaluate_one_epoch(
    model,
    val_dataloader,
    criterion_dict,
    report_interval=10,
):
    model.eval()

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(val_dataloader):
        image, labels = data['image'], data['label']
        outputs = model(image)
        loss = compute_loss(outputs, labels, criterion_dict)
        running_loss += loss.item()

        if i % report_interval == 0:
            last_loss = running_loss / report_interval
            print(f'[{i}/{len(val_dataloader)}]\tLoss: {loss.item():.6f}')
            running_loss = 0.

    return last_loss


def train(config):
    generator = torch.Generator().manual_seed(42)
    transform_fn = transforms.Compose([
        T.Resize(config.input_size),
        ToTensorDict()
    ])
    raw_dataset = RawDataset()
    # create train test split
    patient_dataset = InjuryClassification2DDataset(raw_dataset, lambda x: x)
    train_size = int(0.8 * len(patient_dataset))
    val_size = len(patient_dataset) - train_size
    train_dataset, val_dataset = random_split(patient_dataset, [train_size, val_size], generator=generator)
    # select a subset of the dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    print(f'Loaded {len(train_dataset)} training samples')
    print(f'Loaded {len(val_dataset)} validation samples')

    model = define_model(config.model_name)
    model.to(config.device)

    criterion_dict = {
        'bowel': nn.BCEWithLogitsLoss(),
        'extravasation': nn.BCEWithLogitsLoss(),
        'kidney': nn.CrossEntropyLoss(),
        'liver': nn.CrossEntropyLoss(),
        'spleen': nn.CrossEntropyLoss(),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float('inf')

    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        loss = train_one_epoch(
            model,
            train_dataloader,
            criterion_dict,
            optimizer,
        )
        vloss = evaluate_one_epoch(
            model,
            val_dataloader,
            criterion_dict,
        )
        print(f'Loss: {loss:.6f}, Val Loss: {vloss:.6f}')

        if vloss < best_val_loss:
            best_val_loss = vloss
            torch.save(model.state_dict(), f'../models/{config.model_checkpoint_name}.pth')
            print('Model saved')

    # save the model
