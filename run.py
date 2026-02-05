import sys
import os
import argparse

# Ensure the root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import collect_training_data
from src import train_model
from src import detect_knock_on

def main():
    parser = argparse.ArgumentParser(description="Rugby Knock-On Detection System")
    parser.add_argument('--mode', type=str, choices=['collect', 'train', 'detect'], required=True,
                      help='Mode to run: collect (label data), train (train model), or detect (run detection)')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        print("Starting Data Collection Tool...")
        collect_training_data.main()
    elif args.mode == 'train':
        print("Starting Training...")
        train_model.train()
    elif args.mode == 'detect':
        print("Starting Detection...")
        detect_knock_on.main()

if __name__ == "__main__":
    main()
