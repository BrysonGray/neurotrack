from datetime import datetime
import numpy as np
import os
import pandas as pd
import tifffile as tf
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DATE = datetime.now().strftime("%m-%d-%y")


class StateData(Dataset):
    """
        A custom Dataset class for loading and transforming image data and labels.
        
        Attributes
        ----------
        img_labels : pd.DataFrame
            DataFrame containing image file names and corresponding labels.
        img_dir : str
            Directory where image files are stored.
        transform : callable, optional
            A function/transform to apply to the images.
        target_transform : callable, optional
            A function/transform to apply to the labels.
    """
    
    def __init__(self, labels_file, img_dir, transform=None, target_transform=None, seed=0):
        self.img_labels = pd.read_csv(labels_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.rng = np.random.default_rng(seed)
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # type: ignore
        large_image = tf.imread(img_path) # image with shape (3,21,21,21)
        # image = torch.from_numpy(image[None])
        large_image = torch.from_numpy(large_image)
        shift = self.rng.integers(low=-3, high=4, size=(3,))
        image = large_image[:, 3+shift[0]:18+shift[0], 3+shift[1]:18+shift[1], 3+shift[2]:18+shift[2]]
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def transform(image):
    """
    Apply random transformations to the input image tensor.
    The function performs the following transformations:
    1. Randomly permutes the dimensions of the image tensor.
    2. Randomly flips the image tensor along each of the last three dimensions.
    
    Parameters
    ----------
    image : torch.Tensor
        The input image tensor to be transformed. It is expected to have at least 4 dimensions.
        
    Returns
    -------
    torch.Tensor
        The transformed image tensor with the same shape as the input.
    """
    
    perm = torch.randperm(3) + 1
    image = image.permute([0,*perm])
    if torch.rand(1)>0.5: image = image.flip(-1)
    if torch.rand(1)>0.5: image = image.flip(-2)
    if torch.rand(1)>0.5: image = image.flip(-3)
    
    return image


def transform_spherical_patch(image):
    """
    Apply random transformations to the input image tensor.
    The function randomly flips the image tensor along each of the last two dimensions.
    
    Parameters
    ----------
    image : torch.Tensor
        The input image tensor to be transformed. It is expected to have at least 4 dimensions.
        
    Returns
    -------
    torch.Tensor
        The transformed image tensor with the same shape as the input.
    """
    
    if torch.rand(1)>0.5: image = image.flip(-1)
    if torch.rand(1)>0.5: image = image.flip(-2)
    
    return image


def train_loop(dataloader, model, loss_fn, optimizer, logger=None):
    """
    Train the model for one epoch.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the training data.
    model : torch.nn.Module
        The model to be trained.
    loss_fn : callable
        The loss function to be used.
    optimizer : torch.optim.Optimizer
        The optimizer to be used for updating the model parameters.

    Returns
    -------
    list
        A list of loss values recorded during training.
    """
    
    size = len(dataloader.dataset)
    # positive_count = len(np.where(dataloader.dataset.img_labels.iloc[:,1] > 0.0)[0])
    # negative_count = size - positive_count
    losses = []
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        # out = model(X[:,:3].to(device=DEVICE))
        out = model(X.to(device=DEVICE,dtype=torch.float32))
        out = torch.sigmoid(out.squeeze())
        # out = torch.nn.functional.softmax(out, dim=1)
        # y = torch.nn.functional.one_hot(y, num_classes=3)
        y = y.to(dtype=torch.float, device=DEVICE)
        # weights = torch.where(y > 0.0, positive_count/size, negative_count/size)
        loss = loss_fn(out,y)
        # loss = torch.mean(loss * weights)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X) + len(X)
            losses.append(loss)
            accuracy = ((out > 0.5) == y).type(torch.float).sum().item()
            accuracy = accuracy / len(y) * 100
            print_out = f"Accuracy: {accuracy}, Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]"
            if logger is None:
                print(print_out)
            else:
                logger(print_out)

    return losses

def test_loop(dataloader, model, loss_fn, logger=None):
    """
    Evaluate the model on the test dataset and compute various metrics.
    
    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    model : torch.nn.Module
        The model to be evaluated.
    loss_fn : callable
        The loss function used to compute the loss.
        
    Returns
    -------
    None
        This function prints the test error, accuracy, average loss, precision, and recall.
    """

    size = len(dataloader.dataset)
    # positive_count = len(np.where(dataloader.dataset.img_labels.iloc[:,1] > 0.0)[0])
    # negative_count = size - positive_count
    model.eval()
    num_batches = len(dataloader)
    test_loss, TP, TN, FP, FN = 0,0,0,0,0
    with torch.no_grad():
        for X,y in dataloader:
            # out = model(X[:,:3].to(device=DEVICE))
            out = model(X.to(device=DEVICE,dtype=torch.float32))
            out = torch.sigmoid(out.squeeze())
            y = y.to(dtype=torch.float, device=DEVICE)
            # weights = torch.where(y > 0.0, positive_count/size, negative_count/size)
            loss = loss_fn(out,y)
            # loss = torch.mean(loss * weights)
            test_loss += loss.item()
            threshold = 0.5
            TP_ = ((out > threshold) & (y > 0.0)).type(torch.float).sum().item()
            TN_ = ((out <= threshold) & (y <= 0.0)).type(torch.float).sum().item()
            FP_ = ((out > threshold).sum().item() - TP_)
            FN_ = ((out <= threshold).sum().item() - TN_)
            TP += TP_
            TN += TN_
            FP += FP_
            FN += FN_

    test_loss /= num_batches
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    correct = (TP + TN) / size
    print_out = f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\
              Precision: {precision:>0.3f}, Recall: {recall:>0.3f}"
    if logger is None:
        print(print_out)
    else:
        logger(print_out)
    
    return


def init_dataloader(state_data, batchsize=64):
    """
    Initializes a DataLoader with a WeightedRandomSampler to handle imbalanced datasets.
    
    Parameters
    ----------
    state_data : Dataset
        The dataset containing the image labels and other relevant data.
    batchsize : int, optional
        The number of samples per batch to load (default is 64).
        
    Returns
    -------
    DataLoader
        A DataLoader instance with a WeightedRandomSampler to balance the classes.
    """
    
    nonzero_sample_count = np.sum(state_data.img_labels.iloc[:,1] > 0.0)
    nonzero_weight = 1. / nonzero_sample_count
    zero_sample_count = len(state_data.img_labels) - nonzero_sample_count
    zero_weight = 1. / zero_sample_count
    training_samples_weight = [nonzero_weight if t > 0.0 else zero_weight for t in state_data.img_labels.iloc[:,1]]
    training_sampler = WeightedRandomSampler(training_samples_weight, len(training_samples_weight))
    dataloader = DataLoader(state_data, batch_size=batchsize, sampler=training_sampler)

    return dataloader


def log_training(filename, mode='a'):
    def log_info(info):
        with open(filename, mode) as f:
            f.write(f"{info}\n")
    return log_info


def train(train_dataloader, test_dataloader, out_dir, lr, epochs, classifier, state_dict=None):
    """
    Train a classifier model using the provided dataloaders and parameters.
    
    Parameters
    ----------
    train_dataloader : DataLoader
        DataLoader for the training dataset.
    test_dataloader : DataLoader
        DataLoader for the testing dataset.
    out_dir : str
        Directory where the model checkpoints will be saved.
    lr : float
        Learning rate for the optimizer.
    epochs : int
        Number of epochs to train the model.
    classifier : torch.nn.Module
        The classifier model to be trained.
    state_dict : dict, optional
        State dictionary to load a previously trained model (default is None).
        
    Returns
    -------
    None
    """
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # Create a training log file to track progress
    log_file_path = os.path.join(out_dir, "training_logs.txt")
    logger = log_training(log_file_path, mode='a')
    logger(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger(f"Learning rate: {lr}")
    logger(f"Total epochs: {epochs}\n")

    if state_dict is not None:
        #load a previously trained model
        classifier.load_state_dict(state_dict)

    classifier.train()
    classifier_optimizer = optim.AdamW(classifier.parameters(), lr=lr)
    binary_loss = torch.nn.BCELoss()

    for i,t in enumerate(range(epochs)):
        logger(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, classifier, binary_loss, classifier_optimizer, logger=logger)
        test_loop(test_dataloader, classifier, binary_loss, logger=logger)
        if i%10==0:
            torch.save(classifier.state_dict(), os.path.join(out_dir, f"resnet_classifier_{DATE}_checkpoint-{i}.pt"))
    logger("Done!")
    
    return