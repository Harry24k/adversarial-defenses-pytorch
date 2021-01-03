import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from .datasets import Datasets
    
r"""
Arguments:
    data_name (str): model to train.
    root (str): strength of the attack or maximum perturbation.
    val_info (int or float or list): ratio or index of the validation set in the training set.
    label_filter (dict): label filtering and mapping.
        e.g. `{0:0, 7:1}` means only images with label 0 and 7 are considered.
        Furthermore, label 7 will be changed into 1.
    shuffle_train (bool): True for shuffle train loader.
    transform_train (torchvision.transforms): transform for the training set.
    transform_test (torchvision.transforms): transform for the test set.
    transform_val (torchvision.transforms): transform for the validation set.
    batch_size (int): batch size of the loader.

"""
    
def base_loader(data_name,
                root='./data',
                val_info=None,
                label_filter=None,
                shuffle_train=True,
                shuffle_val=False,
                transform_train=None, 
                transform_test=transforms.ToTensor(), 
                transform_val=transforms.ToTensor(),
                batch_size=128):
    
    if (data_name == "CIFAR10") or (data_name == "CIFAR100") :
        if transform_train is None :
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    if data_name == "TinyImageNet" :
        if transform_train is None :
            transform_train = transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])

    if data_name == "MNIST" :
        if transform_train is None :
            transform_train = transforms.Compose([
                transforms.ToTensor(),
            ])

    # Define Datasets
    data = Datasets(data_name=data_name,
                    root=root,
                    val_info=val_info,
                    label_filter=label_filter,
                    shuffle_train=shuffle_train,
                    shuffle_val=shuffle_val,
                    transform_train=transform_train, 
                    transform_val=transform_val,
                    transform_test=transform_test)

    # Get train_loader, (val_loader), test_loader
    return data.get_loader(batch_size=batch_size)
