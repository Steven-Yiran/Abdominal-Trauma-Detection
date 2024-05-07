import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from datasets.series_injury_classification_dataset import SeriesInjuryClassificationDataset
from model_zoo.series_classification import define_model


def compute_accuracy(outputs, labels):
    """
    Compute multi-label accuracy
    """
    return (outputs > 0.5).eq(labels > 0.5).sum().item() / labels.numel()


def compute_f1(outputs, labels):
    """
    Compute multi-label F1 score
    """
    tp = (outputs > 0.5).eq(labels > 0.5).sum().item()
    fp = (outputs > 0.5).eq(labels < 0.5).sum().item()
    fn = (outputs < 0.5).eq(labels > 0.5).sum().item()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def compute_fn(outputs, labels):
    """
    Compute multi-label F1 score
    """
    fn = (outputs < 0.5).eq(labels > 0.5).sum().item()
    return fn / labels.numel()


def train(config):
    generator = torch.Generator().manual_seed(42)

    # Load the dataset
    patient_dataset = SeriesInjuryClassificationDataset(
        frame_predictions_csv=config.frame_label_path,
        patient_labels_csv=config.train_csv,
        max_series_length=config.max_series_length
    )
    train_size = int(0.8 * len(patient_dataset))
    val_size = len(patient_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        patient_dataset,
        [train_size, val_size],
        generator=generator
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        generator=generator
    )
    print(f'Loaded {len(train_dataset)} training samples')
    print(f'Loaded {len(val_dataset)} validation samples')

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

    # Evaluate the model
    model.eval()

    val_loss = 0
    val_accuracy = 0
    val_f1 = 0
    val_fn = 0
    pbar = tqdm(total=len(val_dataloader), desc='Validation')
    for i, (features, labels) in enumerate(val_dataloader):
        features = features.to(config.device)
        labels = labels.to(config.device)
        outputs = model(features)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_accuracy += compute_accuracy(outputs, labels)
        val_f1 += compute_f1(outputs, labels)
        val_fn += compute_fn(outputs, labels)
        pbar.update(1)

    pbar.close()

    val_loss /= len(val_dataloader)
    val_accuracy /= len(val_dataloader)
    val_f1 /= len(val_dataloader)
    val_fn /= len(val_dataloader)
    
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')
    print(f'Validation F1: {val_f1}')
    print(f'Validation FN: {val_fn}')