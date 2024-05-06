import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import v2 as T
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from model_zoo.injury_classification import define_model
from datasets.raw_dataset import RawDataset
from datasets.injury_classification_2d_dataset import InjuryClassification2DDataset
from utils.transform import ToTensorDict, Resize, RandomCrop, ToPILImage

def compute_loss(outputs, labels, criterion_dict):
    loss = 0.
    for key in outputs.keys():
        loss += criterion_dict[key](outputs[key].squeeze(), labels[key].squeeze())
    return loss


def train_one_epoch(
    epoch,
    model,
    train_dataloader,
    criterion_dict,
    optimizer,
    device,
):
    model.zero_grad()
    model.train()

    running_loss = 0.
    last_loss = 0.

    pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f'Training Epoch {epoch}')
    for i, data in enumerate(train_dataloader):
        image, labels = data['image'], data['label']
        image = image.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}
        optimizer.zero_grad()
        outputs = model(image)
        loss = compute_loss(outputs, labels, criterion_dict)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        pbar.update(1)

    return last_loss


def evaluate_one_epoch(
    model,
    val_dataloader,
    criterion_dict,
    device,
):
    model.eval()

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(val_dataloader):
        image, labels = data['image'], data['label']
        image = image.to(device)
        labels = {key: value.to(device) for key, value in labels.items()}
        outputs = model(image)
        loss = compute_loss(outputs, labels, criterion_dict)
        running_loss += loss.item()

    return last_loss


def train(config):
    generator = torch.Generator().manual_seed(42)
    transform_fn = transforms.Compose([
        ToPILImage(),
        Resize(config.input_size),
        RandomCrop(config.input_size),
        ToTensorDict(),
    ])
    raw_dataset = RawDataset(
        csv_path=config.train_csv,
        image_dir=config.img_dir,
    )
    # create train test split
    patient_dataset = InjuryClassification2DDataset(raw_dataset, transform=transform_fn)
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
        loss = train_one_epoch(
            epoch,
            model,
            train_dataloader,
            criterion_dict,
            optimizer,
            config.device,
        )
        vloss = evaluate_one_epoch(
            model,
            val_dataloader,
            criterion_dict,
            config.device,
        )
        print(f'Loss: {loss:.6f}, Val Loss: {vloss:.6f}')

        if vloss < best_val_loss:
            best_val_loss = vloss
            torch.save(model.state_dict(), f'../models/{config.model_checkpoint_name}.pth')
            print('Model saved')