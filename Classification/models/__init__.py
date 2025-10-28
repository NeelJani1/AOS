# models/__init__.py

# --- Step 1: Import models from their correct files ---

# Import ImageNet-style ResNets from ResNet.py
from .ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

# Import VGG models from VGG.py
from .VGG import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn

# Import CIFAR/Tiny-style ResNets from ResNet_s.py
from .ResNets import resnet20s, resnet32s, resnet44s, resnet56s, resnet110s, resnet1202s


# --- Step 2: Create the master dictionary ---
# This is the 'model_dict' that utils.py imports.
model_dict = {
    # From ResNet.py
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    
    # From VGG.py
    "vgg11": vgg11,
    "vgg11_bn": vgg11_bn,
    "vgg13": vgg13,
    "vgg13_bn": vgg13_bn,
    "vgg16": vgg16,
    "vgg16_bn": vgg16_bn,
    "vgg19": vgg19,
    "vgg19_bn": vgg19_bn,
    
    # From ResNet_s.py
    "resnet20s": resnet20s,
    "resnet32s": resnet32s,
    "resnet44s": resnet44s,
    "resnet56s": resnet56s,
    "resnet110s": resnet110s,
    "resnet1202s": resnet1202s,
}

# --- Step 3: Export the model_dict variable ---
# This line makes `from models import model_dict` work in utils.py
__all__ = ["model_dict"]