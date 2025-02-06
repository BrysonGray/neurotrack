import argparse
import os
from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).parents[1]))
import models
from solvers import branch_classifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', type=str, help='Source directory containing label annotations as csv files and input images folder (observations).')
    parser.add_argument('-o','--out', type=str, help="Path to output directory.")
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Optimizer learning rate.')
    parser.add_argument('-N', '--epochs', type=int, default=15, help='Number of training epochs.')

    args = parser.parse_args()
    source = args.source
    out_dir = args.out
    lr = args.learning_rate
    epochs = args.epochs
    
    source_list = os.listdir(source)
    training_annotations_file = [f for f in source_list if 'training_annotations' in f][0]
    training_annotations_file = os.path.join(source, training_annotations_file)
    if not os.path.exists(training_annotations_file):
        raise FileNotFoundError("Source directory must contain a csv file with \"training_annotations\" in the filename,\
                                but none was found.")
    test_annotations_file = [f for f in source_list if 'test_annotations' in f][0]
    test_annotations_file = os.path.join(source, test_annotations_file)
    if not os.path.exists(test_annotations_file):
        raise FileNotFoundError("Source directory must contain a csv file with \"test_annotations\" in the filename,\
                                but none was found")
    img_dir = os.path.join(source, 'observations')
    if not os.path.exists(img_dir):
        raise FileNotFoundError("Source directory must contain a folder named \"observations\",\
                                but none was found.")

    transform = branch_classifier.transform # random permutation and flip
    training_data = branch_classifier.StateData(annotations_file=training_annotations_file,
                            img_dir=img_dir,
                            transform=transform)
    test_data = branch_classifier.StateData(annotations_file=test_annotations_file,
                            img_dir=img_dir)

    training_dataloader = branch_classifier.init_dataloader(training_data)
    test_dataloader = branch_classifier.init_dataloader(test_data)
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    classifier = models.ResNet(models.ResidualBlock, [3, 4, 6, 3], num_classes=1)
    classifier = classifier.to(device=DEVICE, dtype=dtype)

    branch_classifier.train(training_dataloader, test_dataloader, out_dir, lr, epochs, classifier, state_dict=None)
    
    return

if __name__ == "__main__":
    main()