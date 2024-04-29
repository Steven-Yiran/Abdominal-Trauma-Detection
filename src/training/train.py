import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset

from model_zoo.models import define_model
from dataset.dataset import PatientDataset
from dataset.transform import Rescale, RandomCrop, ToTensorDict

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
    report_interval,
):
    model.zero_grad()
    model.train()

    running_loss = 0.
    last_loss = 0.

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

    return last_loss


def train(config):
    transform_fn = transforms.Compose([
        Rescale(config.input_size + config.crop_size),
        RandomCrop(config.crop_size),
        ToTensorDict()
    ])
    train_dataset = PatientDataset(
        csv_file=config.train_csv,
        img_dir=config.img_dir,
        transform=transform_fn,
    )
    # select a subset of the dataset
   # train_dataset = Subset(train_dataset, range(2))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    print(f'Loaded {len(train_dataset)} training samples')

    model = define_model(config.model_name)
    model.double()

    criterion_dict = {
        'bowel': nn.BCEWithLogitsLoss(),
        'extravasation': nn.BCEWithLogitsLoss(),
        'kidney': nn.CrossEntropyLoss(),
        'liver': nn.CrossEntropyLoss(),
        'spleen': nn.CrossEntropyLoss(),
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.num_epochs):
        print(f'Epoch {epoch+1}/{config.num_epochs}')
        loss = train_one_epoch(
            model,
            train_dataloader,
            criterion_dict,
            optimizer,
            config.report_interval,
        )
        print(f'Loss: {loss:.6f}')