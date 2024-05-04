import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from model_zoo.models import define_model
from datasets.sample_dataset import SamplePatientDataset
from utils.transform import Rescale, RandomCrop, ToTensor

def train_one_epoch(
    model,
    train_dataloader,
    criterion,
    optimizer,
    report_interval,
):
    """
    Perform training on a single epoch.
    """
    model.zero_grad()
    model.train()

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(train_dataloader):
        inputs, label = data['image'], data['label']
        # zero gradient for every batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % report_interval == 0:
            last_loss = running_loss / report_interval
            print(f'[{i}/{len(train_dataloader)}]\tLoss: {loss.item():.6f}')
            running_loss = 0.

    return last_loss


def evaluate(
    model,
    vali_dataloader,
    criterion,
    num_classes=13,
):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    running_loss = 0.
    correct = 0

    with torch.no_grad():
        for data in vali_dataloader:
            inputs, label = data['image'], data['label']
            outputs = model(inputs)
            loss = criterion(outputs, label)
            running_loss += loss.item()
            # threshold the output
            preds = (outputs > 0.5).float()
            correct += (preds == label).sum().item()
    
    avg_loss = running_loss / len(vali_dataloader.dataset)
    accuracy = correct / (len(vali_dataloader.dataset) * num_classes)

    return avg_loss, accuracy


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
    # split the dataset into training and validation
    train_size = int(0.8 * len(train_dataset))
    vali_size = len(train_dataset) - train_size
    train_dataset, vali_dataset = random_split(
        train_dataset, [train_size, vali_size]
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    vali_dataloader = DataLoader(
        vali_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    print('Train dataset size: {}'.format(len(train_dataset)))
    print('Validation dataset size: {}'.format(len(vali_dataset)))

    # Define the model
    model = define_model(
        model_name=config.model_name,
        num_classes=config.num_classes
    )

    model.double()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(config.num_epochs):
        print('Epoch {}'.format(epoch + 1))
        avg_loss = train_one_epoch(
            model=model,
            train_dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            report_interval=5,
        )

        avg_vloss, accuracy = evaluate(
            model=model,
            vali_dataloader=vali_dataloader,
            criterion=criterion,
        )
        
        print(f'LOSS train: {avg_loss:.6f}, validation: {avg_vloss:.6f}, accuracy: {accuracy:.6f}')


