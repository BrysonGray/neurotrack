"""
Train branch classifier model.
"""

import argparse
import os
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader
script_path = Path(os.path.abspath(__file__))
parent_dir = script_path.parent.parent  # Go up two levels
sys.path.append(str(parent_dir))
import models
from solvers import branch_classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

def main():
    """
    Main function to train the branch classifier model.

    Arguments
    ---------
    -s, --source: Source directory containing labels as csv files and input images folder (observations).
    -o, --out: Path to output directory.
    -l, --learning_rate: Optimizer learning rate.
    -N, --epochs: Number of training epochs.
    -w --weights: pretrained model weights 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, help='Source directory containing labels as csv files and input images folder (observations).')
    parser.add_argument('-o','--out', type=str, help="Path to output directory.")
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('-N', '--epochs', type=int, default=15, help='Number of training epochs.')
    parser.add_argument('-w', '--weights', type=str, default=None, help='pretrained model weights')

    args = parser.parse_args()
    source = args.source
    out_dir = args.out
    lr = args.learning_rate
    epochs = args.epochs
    weights = args.weights

    source_list = os.listdir(source)
    training_labels_file = [f for f in source_list if 'training_labels' in f][0]
    training_labels_file = os.path.join(source, training_labels_file)
    if not os.path.exists(training_labels_file):
        raise FileNotFoundError("Source directory must contain a csv file with `training_labels` in the filename,\
                                but none was found.")
    test_labels_file = [f for f in source_list if 'test_labels' in f][0]
    test_labels_file = os.path.join(source, test_labels_file)
    if not os.path.exists(test_labels_file):
        raise FileNotFoundError("Source directory must contain a csv file with `test_labels` in the filename,\
                                but none was found")
    img_dir = [d for d in source_list if 'observations' in d][0]
    img_dir = os.path.join(source, img_dir)
    # img_dir = os.path.join(source, 'observations')
    if not os.path.exists(img_dir):
        raise FileNotFoundError("Source directory must contain a folder named `observations`,\
                                but none was found.")

    transform = branch_classifier.transform # random permutation and flip
    training_data = branch_classifier.StateData(labels_file=training_labels_file,
                            img_dir=img_dir,transform=transform)
    test_data = branch_classifier.StateData(labels_file=test_labels_file,
                            img_dir=img_dir)

    # training_dataloader = branch_classifier.init_dataloader(training_data)
    # test_dataloader = branch_classifier.init_dataloader(test_data)
    batchsize=50
    training_dataloader = DataLoader(training_data, batch_size=batchsize)
    test_dataloader = DataLoader(test_data, batch_size=batchsize)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # classifier = models.ResNet2D(models.ResidualBlock2D, [3, 4, 6, 3], in_channels=36, num_classes=1)
    classifier = models.ResNet3D(models.ResidualBlock3D, [3, 4, 6, 3], in_channels=3, num_classes=1)
    classifier = classifier.to(device=DEVICE, dtype=dtype)

    if weights is not None:
        state_dict = torch.load(weights, map_location=DEVICE)
        classifier.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {weights}")

    branch_classifier.train(training_dataloader, test_dataloader, out_dir, lr, epochs, classifier, state_dict=None)
    
    return

if __name__ == "__main__":
    main()