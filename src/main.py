import os
import sys
import argparse
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("--train-classifier", action="store_true", help="Train the injury classifier")
    parser.add_argument("--frame-inference", action="store_true", help="Inference on frame-level injuries and organ probabilities")

    return parser.parse_args()


class Config:

    # Data
    train_csv = "../data/train.csv"
    img_dir = "../data"
    class_names = [
        "bowel_healthy",
        "extravasation_healthy",
        "kidney_healthy",
        "kidney_low",
        "kidney_high",
        "liver_healthy",
        "liver_low",
        "liver_high",
        "spleen_healthy",
        "spleen_low",
        "spleen_high"
    ]

    # Preprocessing
    input_size = 224
    num_workers = 4

    # Model
    model_name = "efficientnet_b0"
    num_classes = 13

    # Training
    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001
    model_checkpoint_name = "../models/model"

    # Inference
    segmentations_csv = "../data/df_images_train_with_seg.csv"
    frame_label_path = "../results/frame_labels.csv"


def main():
    config = Config()

    args = parse_args()

    if args.train_classifier:
        from training.injury_classification import train

        train(config)
    
    if args.frame_inference:
        from data_preparation.rnn_training_data import generate

        generate(config)

if __name__ == "__main__":
    main()

