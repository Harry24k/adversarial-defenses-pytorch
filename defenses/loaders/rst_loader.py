import numpy as np

from torch.utils.data import DataLoader
import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from .datasets import Datasets

from .rst_datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS

r"""
Arguments:
    data_name (str): model to train.
    root (str): strength of the attack or maximum perturbation.
    transform_train (torchvision.transforms): transform for the training set.
    transform_test (torchvision.transforms): transform for the test set.
    batch_size (int): batch size of the loader.

"""
def rst_loader(data_name,
               root='./data',
               val_info=None,
               transform_train=None, 
               transform_test=transforms.ToTensor(), 
               transform_val=transforms.ToTensor(),
               shuffle_val=False,
               batch_size=128):
    
    if (data_name == "CIFAR10") or (data_name == "CIFAR100"):
        if transform_train is None :
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            
    else:
        raise ValueError("%s is not supported in RST"%(data_name))
    
    # Define Datasets
    data = Datasets(data_name=data_name,
                    root=root,
                    val_info=val_info,
                    #label_filter=label_filter,
                    shuffle_train=True,
                    shuffle_val=shuffle_val,
                    transform_train=transform_train, 
                    transform_val=transform_val,
                    transform_test=transform_test)

    # Get train_loader, test_loader
    if val_info is None:
        _, test_loader = data.get_loader(batch_size=batch_size)
    else:
        _, val_loader, test_loader = data.get_loader(batch_size=batch_size)

    # Get train_loader with semisup
    unsup_fraction=0.5

    trainset = SemiSupervisedDataset(base_dataset=data_name.lower(),
                                     add_svhn_extra=False,
                                     root=root, train=True,
                                     download=True, transform=transform_train,
                                     aux_data_filename='ti_500K_pseudo_labeled.pickle',
                                     add_aux_labels=True,
                                     aux_take_amount=None)

    # num_batches=50000 enforces the definition of an "epoch" as passing through 50K
    # datapoints
    # TODO: make sure that this code works also when trainset.unsup_indices=[]
    train_batch_sampler = SemiSupervisedSampler(
        data.train_idx, trainset.unsup_indices,
        batch_size, unsup_fraction,
        num_batches=int(np.ceil(50000 / batch_size)))

    train_loader = DataLoader(trainset, batch_sampler=train_batch_sampler)

    if val_info is None:
        return train_loader, test_loader
    else:
        return train_loader, val_loader, test_loader
    
