from .loaders.base_loader import base_loader
from .loaders.rst_loader import rst_loader

def get_loader(name):
    if name == "Base":
        return base_loader
        
    elif name == "RST":
        return rst_loader
        
    else:
        raise ValueError("Not valid trainer name.")