from enum import Enum
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


class DatasetType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


random_seed = 42
val_size = 1000

"""
 composition = albu.Compose([albu.HorizontalFlip(p = 0.5),
                                    albu.VerticalFlip(p = 0.5),
                                    albu.GridDistortion(p = 0.2),
                                    albu.ElasticTransform(p = 0.2)])
"""


class AugmentedCIFAR100(data.Dataset):
    # TODO: add more processing and augmentation
    def __init__(self,
                 dataset_type: DatasetType):
        super(AugmentedCIFAR100, self).__init__()
        torch.manual_seed(random_seed)
        self.dataset_type = dataset_type
        self.base_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if self.dataset_type is DatasetType.TEST:
            self.data = torchvision.datasets.CIFAR100(root='./data/CIFAR100',
                                                      train=False,
                                                      download=True,
                                                      transform=self.base_transform)
        elif self.dataset_type is self.dataset_type.TRAIN:
            training_transformations = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomRotation(degrees=15),
                 transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                 transforms.RandomApply(transforms=
                 [

                     transforms.ColorJitter(hue=0.5,
                                            brightness=0.3,
                                            contrast=0.5,
                                            saturation=0.5)
                 ], p=0.05),
                 transforms.RandomApply(transforms=
                 [
                     transforms.GaussianBlur(kernel_size=(3, 5),
                                             sigma=(0.05, 2)),

                 ], p=0.05),
                 transforms.RandomGrayscale(0.1)
                 ])
            original_train_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100',
                                                                   train=True,
                                                                   download=True,
                                                                   transform=training_transformations)
            train_size = len(original_train_dataset) - val_size
            train_set, _ = random_split(original_train_dataset, [train_size, val_size])
            self.data = train_set
        elif self.dataset_type is self.dataset_type.VAL:
            original_train_dataset = torchvision.datasets.CIFAR100(root='./data/CIFAR100',
                                                                   train=True,
                                                                   download=True,
                                                                   transform=self.base_transform)
            train_size = len(original_train_dataset) - val_size
            _, val_set = random_split(original_train_dataset, [train_size, val_size])
            self.data = val_set
        else:
            raise NotImplementedError('UNHANDLED dataset_type')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.__getitem__(idx)

    @staticmethod
    def create_loader(dataset_type: DatasetType,
                      batch_size: int) -> data.DataLoader:
        dataset = AugmentedCIFAR100(dataset_type)
        if dataset_type is DatasetType.TRAIN:
            loader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=4,
                                     pin_memory=True)
        elif dataset_type is DatasetType.TEST:
            loader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=4)
        elif dataset_type is DatasetType.VAL:
            loader = data.DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     num_workers=4)
        else:
            raise NotImplementedError('UNHANDLED dataset_type')
        return loader
