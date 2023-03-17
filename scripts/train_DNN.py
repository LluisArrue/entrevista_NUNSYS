from argparse import ArgumentParser
import datetime
from os import makedirs
from os.path import join
from typing import Optional, Union, Tuple
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout: float = 0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out


def get_dataset(X_path: str, y_path: str, get_shape: bool = False) -> Union[TensorDataset, Tuple[TensorDataset, int]]:
    """
    Load data from numpy files, create tensor datasets and return them along with the input shape.

    Args:
        X_path (str): Path to the numpy file containing the input features.
        y_path (str): Path to the numpy file containing the target labels.
        get_shape (bool, optional): If True, also return the input shape. Defaults to False.

    Returns:
        Union[TensorDataset, Tuple[TensorDataset, int]]: If `get_shape` is False, return a TensorDataset object containing
        the input features and target labels as tensors. If `get_shape` is True, also return the input shape as an integer.
    """
    X = np.load(X_path)
    y = np.load(y_path)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)
    if get_shape:
        return TensorDataset(X_tensor, y_tensor), X.shape[-1]
    else:
        return TensorDataset(X_tensor, y_tensor)


def compute_acc(pred: torch.Tensor, real: torch.Tensor) -> int:
    """
    Computes the accuracy of a binary classifier by comparing its predictions to the ground truth labels.

    Args:
        pred (torch.Tensor): A tensor containing the predicted binary labels.
        real (torch.Tensor): A tensor containing the true binary labels.

    Returns:
        int: The number of correct predictions.

    """
    return ((pred > .5) == real).sum().item()


def train_model(X_train: str, y_train: str, num_epochs: int, hidden_size: int, X_test: Optional[str] = None,
                y_test: Optional[str] = None, X_val: Optional[str] = None, y_val: Optional[str] = None, bs: int = 32,
                lr: float = 0.001, dropout: float = 0.5, outpath: Optional[str] = None) -> torch.nn.Module:
    """
    Trains a neural network model using the provided training data.

    Args:
        X_train (str): Path to the npy file containing the training data.
        y_train (str): Path to the npy file containing the labels for the training data.
        num_epochs (int): Number of training epochs.
        hidden_size (int): Number of units in the hidden layer.
        X_test (Optional[str]): Path to the npy file containing the test data.
        y_test (Optional[str]): Path to the npy file containing the labels for the test data.
        X_val (Optional[str]): Path to the npy file containing the validation data.
        y_val (Optional[str]): Path to the npy file containing the labels for the validation data.
        bs (int): Batch size for training the model.
        lr (float): Learning rate for the optimizer.
        dropout (float): Dropout probability for regularization.

    Returns:
        nn.Module: The trained neural network model.
    """
    run_id = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    train_dataset, input_size = get_dataset(X_train, y_train, True)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)

    if X_val and y_val:
        val_dataset = get_dataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=bs)
    else:
        val_dataset = None
        val_loader = None

    if X_test and y_test:
        test_dataset = get_dataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=bs)
    else:
        test_dataset = None
        test_loader = None

    print('Data loaded')

    model = Net(input_size, hidden_size, 1, dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    writer = SummaryWriter(join('runs', run_id))

    writer.add_graph(model, train_dataset[0][0].reshape(1, -1))

    print('Training started...')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += compute_acc(outputs, labels)

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = train_acc / len(train_loader)

        writer.add_scalar('train/loss', epoch_loss, epoch + 1)
        writer.add_scalar('train/accuracy', epoch_acc, epoch + 1)

        if not val_dataset:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss}")
            continue

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch

                y_pred = model(inputs)

                val_loss += criterion(y_pred, labels).item()
                val_acc += compute_acc(y_pred, labels)

        val_loss /= len(val_loader)
        val_acc /= len(val_dataset)

        writer.add_scalar('validation/loss', val_loss, epoch + 1)
        writer.add_scalar('validation/accuracy', val_acc, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val "
            f"Acc: {val_acc:.4f}")

    print('Training finished!')

    if outpath:
        makedirs(outpath, exist_ok=True)
        model_path = join(outpath, 'model_' + run_id + '.pt')
        torch.save(
            model.state_dict(),
            model_path
        )
        print('Model saved at', model_path)

    if test_loader is None or test_dataset is None:
        return model

    print('Evaluating model...')

    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch

            y_pred = model(inputs)

            test_loss += criterion(y_pred, labels).item()
            test_acc += compute_acc(y_pred, labels)

    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_dataset)

    writer.add_hparams({
        'lr': lr,
        'bsize': bs,
        'num epochs': num_epochs,
        'input size': input_size,
        'hidden size': hidden_size,
        'train length': len(train_loader),
        'test length': len(test_loader) if test_loader is not None else 0,
        'val length': len(val_loader) if test_loader is not None else 0,
        'dropout': dropout
    }, {
        'test/loss': test_loss,
        'test/acc': test_acc
    })
    print(f'Test loss: {test_loss:.4f} Test accuracy: {test_acc:.4f}')

    return model


def get_parser():
    parser = ArgumentParser(description='Train a neural network model.')

    # Required arguments
    parser.add_argument('-x', '--x-train', required=True, type=str, help='Path to training data')
    parser.add_argument('-y', '--y-train', required=True, type=str, help='Path to training labels')
    parser.add_argument('-n', '--num-epochs', required=True, type=int, help='Number of epochs for training')
    parser.add_argument('--hidden', '--hidden-size', required=True, type=int, help='Number of units in the hidden layer')

    # Optional arguments with default values
    parser.add_argument('--xt', '--x-test', type=str, default=None, help='Path to test data')
    parser.add_argument('--yt', '--y-test', type=str, default=None, help='Path to test labels')
    parser.add_argument('--xv', '--x-val', type=str, default=None, help='Path to validation data')
    parser.add_argument('--yv', '--y-val', type=str, default=None, help='Path to validation labels')
    parser.add_argument('--bs', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability for regularization')
    parser.add_argument('-o', '--output', type=str, default=None, help='path to save the model')

    return parser


def main():
    args = get_parser().parse_args()
    train_model(args.x_train, args.y_train, args.num_epochs, args.hidden, args.xt, args.yt,
                args.xv, args.yv, args.bs, args.lr, args.dropout, args.output)


if __name__ == '__main__':
    main()
