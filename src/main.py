import torch
import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("--train-classifier", action="store_true", help="Train the injury classifier")
    parser.add_argument("--frame-inference", action="store_true", help="Inference on frame-level injuries and organ probabilities")
    parser.add_argument("--train-rnn", action="store_true", help="Train the RNN")
    parser.add_argument("--end-to-end", action="store_true", help="Train the classifier, generate frame-level predictions, and train the RNN")

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

    # Sampling
    strategy = "Uniform" # Random or OrganBased or Sequential or Uniform
    threshold = 0.5
    max_frame_per_patient = 50

    # Model
    model_name = "efficientnet_b0"
    num_classes = 13

    # Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 1
    batch_size = 32
    learning_rate = 0.001
    model_checkpoint_name = "../models/model"

    # Training - RNN
    input_dim = 15
    hidden_dim = 128
    num_layers = 2
    output_dim = 11
    max_series_length = 50

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

    if args.train_rnn:
        from training.series_classification import train

        train(config)

    if args.end_to_end:
        from training.injury_classification import train as train_classifier
        from data_preparation.rnn_training_data import generate
        from training.series_classification import train as train_rnn

        train_classifier(config)
        generate(config)
        train_rnn(config)

if __name__ == "__main__":
    main()

