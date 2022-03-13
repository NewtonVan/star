import numpy as np
import torch
import argparse

from typing import Tuple
from torch.utils.data import Dataset, DataLoader


def load_data(label: str) -> Tuple[np.ndarray, np.ndarray]:
    if label == 'train':
        return np.load('x_train.npy'), np.load('y_train.npy')
    else:
        return np.load('x_test.npy'), np.load('y_test.npy')


class StarDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.past = x
        self.future = y

    def __len__(self) -> int:
        return self.past.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.past[idx]), torch.from_numpy(self.future[idx])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=300, type=int, help='epochs')
    parser.add_argument('--batch', default=100, type=int, help='batch size')
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch

    x_train, y_train = load_data('train')
    x_test, y_test = load_data('test')
    train_dataset = StarDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for x_train_batch, y_train_batch in train_loader:
        print('x_train', x_train.shape)
        print('y_train', y_train.shape)
