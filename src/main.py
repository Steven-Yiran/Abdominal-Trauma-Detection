import os
import sys
import argparse
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("--train-classifier", action="store_true", help="Train the injury classifier")
    parser.add_argument("--inference-frame-injuries", action="store_true", help="Inference on frame-level injuries")

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
    results_csv = "../results/results.csv"


def main():
    config = Config()

    args = parse_args()

    if args.train_classifier:
        from training.injury_classification import train

        train(config)
    
    if args.inference_frame_injuries:
        from inference.frame_inference import infer_frame_injuries

        infer_frame_injuries(config)

if __name__ == "__main__":
    main()

