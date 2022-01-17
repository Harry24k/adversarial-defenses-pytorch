import torchvision.transforms as transforms

from .datasets.base import Datasets

from .models.lenet import LeNet
from .models.mnist_ates import MNIST_ATES
from .models.mnist_dat import MNIST_DAT
from .models.mnist_fast import MNIST_Fast
from .models.preactresnet import PreActBlock, PreActResNet
from .models.resnet import ResBasicBlock, ResNet
from .models.densenet import DenseNet, Bottleneck
from .models.vgg import VGG
from .models.wideresnet import WideResNet

from .cuda import fix_randomness, fix_gpu
    
def load_model(model_name, n_classes):
    if model_name == "LeNet":
        return LeNet(n_classes)
    
    elif model_name == "MNIST_ATES":
        return MNIST_ATES(n_classes)
    
    elif model_name == "MNIST_DAT":
        return MNIST_DAT(n_classes)
    
    elif model_name == "MNIST_Fast":
        return MNIST_Fast(n_classes)
    
    elif model_name == "WRN28-10":
        model = WideResNet(depth=28, num_classes=n_classes, 
                           widen_factor=10, dropRate=0.3)
        
    elif model_name == "WRN28-10-D0":
        model = WideResNet(depth=28, num_classes=n_classes,
                           widen_factor=10, dropRate=0.0)
        
    elif model_name == "WRN34-10":
        model = WideResNet(depth=34, num_classes=n_classes,
                           widen_factor=10, dropRate=0.3)
        
    elif model_name == "WRN34-10-D0":
        model = WideResNet(depth=34, num_classes=n_classes,
                           widen_factor=10, dropRate=0.0)
        
    elif model_name == "PRN18":
        model = PreActResNet(PreActBlock, num_blocks=[2,2,2,2],
                             num_classes=n_classes)
        
    elif model_name == "ResNet10":
        model = ResNet(ResBasicBlock, [1, 1, 1, 1], n_classes, in_channels=1)
        
    elif model_name == "ResNet18":
        model = ResNet(ResBasicBlock, [2, 2, 2, 2], n_classes)
        
    elif model_name == "ResNet34":
        model = ResNet(ResBasicBlock, [3, 4, 6, 3], n_classes)
        
    elif model_name == "ResNet50":
        model = ResNet(ResBasicBlock, [3, 4, 6, 3], n_classes)
        
    elif model_name == "ResNet101":
        model = ResNet(ResBasicBlock, [3, 4, 23, 3], n_classes)
    
    elif model_name == "ResNet152":
        model = ResNet(ResBasicBlock, [3, 8, 36, 3], n_classes)
        
    elif model_name == "DenseNet121":
        model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

    elif model_name == "DenseNet169":
        model = DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

    elif model_name == "DenseNet201":
        model = DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

    elif model_name == "DenseNet161":
        model = DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)
        
    elif model_name == "VGG11":
        model = VGG('VGG11', n_classes)
        
    elif model_name == "VGG13":
        model = VGG('VGG13', n_classes)
        
    elif model_name == "VGG16":
        model = VGG('VGG16', n_classes)
        
    elif model_name == "VGG19":
        model = VGG('VGG19', n_classes)
        
    else:
        raise ValueError('Invalid model name.')
        
    print(model_name, "is loaded.")
        
    return model