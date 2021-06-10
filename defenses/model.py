import torch.nn as nn

from .models.lenet import LeNet
from .models.mnist_ates import MNISTATES
from .models.mnist_custom import MNISTLARGE
from .models.mnist_dat import MNISTDAT
from .models.mnist_fast import MNISTFast

from .models.preact_resnet import PreActBlock, PreActResNet
from .models.resnet import ResBasicBlock, ResNet
from .models.densenet import DenseNet, Bottleneck
from .models.vgg import VGG
from .models.wide_resnet import WideResNet

from .models.normalize import Normalize, Identity

def get_model(name, num_classes, fc_input_dim_scale=1):
    
    # MNIST w/o Normalize Layer
    if name == "LeNet":
        return MNIST_NET(num_classes)
    
    if name == "MNISTATES":
        return MNISTATES(num_classes)
    
    if name == "MNISTLARGE":
        return MNISTLARGE(num_classes)
    
    if name == "MNISTDAT":
        return MNISTDAT(num_classes)
    
    if name == "MNISTFast":
        return MNISTFast(num_classes)
    
    # CIFAR w/ Normalize Layer
    norm = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    if name == "WRN28":
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN28-D0":
        model = WideResNet(depth=28, num_classes=num_classes, widen_factor=10, dropRate=0.0, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN34":
        model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN34-D0":
        model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10, dropRate=0.0, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN40":
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=10, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN40-2":
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0.3, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "WRN40-W2-D0":
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, dropRate=0, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "PRN18":
        model = PreActResNet(PreActBlock, num_blocks=[2,2,2,2], num_classes=num_classes, fc_input_dim_scale=fc_input_dim_scale)
        
    if name == "ResNet18":
        model = ResNet(ResBasicBlock, [2, 2, 2, 2], num_classes)
        
    if name == "ResNet34":
        model = ResNet(ResBasicBlock, [3, 4, 6, 3], num_classes)
        
    if name == "ResNet50":
        model = ResNet(ResBottleneck, [3, 4, 6, 3], num_classes)
        
    if name == "ResNet101":
        model = ResNet(ResBottleneck, [3, 4, 23, 3], num_classes)
    
    if name == "ResNet152":
        model = ResNet(ResBottleneck, [3, 8, 36, 3], num_classes)
        
    if name == "DenseNet121":
        model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

    if name == "DenseNet169":
        model = DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

    if name == "DenseNet201":
        model = DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

    if name == "DenseNet161":
        model = DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

    if name == "DenseNetCIFAR":
        model = DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)
        
    if name == "VGG11":
        model = VGG('VGG11', num_classes)
        
    if name == "VGG13":
        model = VGG('VGG13', num_classes)
        
    if name == "VGG16":
        model = VGG('VGG16', num_classes)
        
    if name == "VGG19":
        model = VGG('VGG19', num_classes)
    
    print(name, "is loaded.")
        
    return nn.Sequential(norm, model)