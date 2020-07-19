import torchvision.utils
import torchvision.datasets as dsets
import torchvision.transforms as transforms

from torchhk import Datasets

import numpy as np

print("Loader : Small Dataset 2")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def get_loader(data_name="CIFAR10", shuffle_train=True,
               transform_train=transform_train,
               transform_test=transform_test, batch_size=128):
    
    aa = np.arange(0,50000)
    np.random.shuffle(aa)
    
    
    # Define Datasets
    data = Datasets(data_name=data_name,
                    val_idx=aa[:30000],
                    shuffle_train=shuffle_train,
                    transform_train=transform_train,
                    transform_test=transform_test)

    # Get train_loader, test_loader
    train_loader, val_loader, test_loader = data.get_loader(batch_size=batch_size)
    
    return train_loader, test_loader