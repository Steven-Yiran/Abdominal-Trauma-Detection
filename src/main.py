import os
import sys
import argparse
import matplotlib.pyplot as plt

from dataset.dataset import SamplePatientDataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("-d", "--data", default='..'+os.sep+'data', help="Location of data file")

    return parser.parse_args()

class Config:

    # Data
    train_csv = "../data/train.csv"
    img_dir = "../data"

    # Preprocessing
    input_size = 224
    crop_size = 224

    # Model
    model_name = "efficientnet_b0"
    num_classes = 13

    # Training
    num_epochs = 5
    batch_size = 32
    learning_rate = 0.001
    report_interval = 10

def main():
    config = Config()

    args = parse_args()

    from training.train import train

    train(config)

    # TODO: run LSTM


if __name__ == "__main__":
    main()

