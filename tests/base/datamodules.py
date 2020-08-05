import os
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

from pytorch_lightning.core.datamodule import LightningDataModule
from tests.base.datasets import TrialMNIST
from torchvision.datasets import MNIST
from torch.utils.data.distributed import DistributedSampler


class TrialMNISTDataModule(LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.non_picklable = None

    def prepare_data(self):
        TrialMNIST(self.data_dir, train=True, download=True)
        TrialMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            mnist_full = TrialMNIST(root=self.data_dir, train=True, num_samples=64, download=True)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [128, 64])
            self.dims = self.mnist_train[0][0].shape

        if stage == 'test' or stage is None:
            self.mnist_test = TrialMNIST(root=self.data_dir, train=False, num_samples=32, download=True)
            self.dims = getattr(self, 'dims', self.mnist_test[0][0].shape)

        self.non_picklable = lambda x: x**2

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size=32, dist_sampler=False):
        super().__init__()

        self.dist_sampler = dist_sampler
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        self.dims = (1, 28, 28)

    def prepare_data(self):
        # download only
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        # TODO: need to split using random_split once updated to torch >= 1.6
        if stage == 'fit' or stage is None:
            self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transforms)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        dist_sampler = None
        if self.dist_sampler:
            dist_sampler = DistributedSampler(self.mnist_train, shuffle=False)

        return DataLoader(
            self.mnist_train, batch_size=self.batch_size, sampler=dist_sampler, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=False)
