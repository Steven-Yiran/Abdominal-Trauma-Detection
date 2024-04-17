import os
import sys
import argparse
import matplotlib.pyplot as plt

from preprocess import SamplePatientDataset

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="My program description")
    parser.add_argument("-d", "--data", default='..'+os.sep+'data', help="Location of data file")

    return parser.parse_args()

def main():
    args = parse_args()
    # Run script from location of main.py
    os.chdir(sys.path[0])

    # Load and preprocess the data from the data directory
    dataset = SamplePatientDataset(
        csv_file=f"{args.data}/train.csv",
        img_dir=f"{args.data}"
    )

    print(dataset[0])

    ## Do something with the data




if __name__ == "__main__":
    main()

