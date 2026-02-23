import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights

# --- Custom Blocks ---
class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        
        self.dcn = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)
        
    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)

def _replace_layers_with_dcn(model):
    """
    Replaces 3x3 convolutions in late layers (layer2, layer3, layer4) with Deformable Convs
    """
    layers_to_modify = ['layer2', 'layer3', 'layer4']
    for layer_name in layers_to_modify:
        layer = getattr(model, layer_name)
        for i, block in enumerate(layer.children()):
            if hasattr(block, 'conv1') and block.conv1.kernel_size == (3, 3):
                new_conv = DeformableBlock(
                    block.conv1.in_channels, 
                    block.conv1.out_channels, 
                    kernel_size=3, 
                    stride=block.conv1.stride, 
                    padding=block.conv1.padding, 
                    bias=False
                )
                # Copy standard weights
                new_conv.dcn.weight.data = block.conv1.weight.data
                block.conv1 = new_conv
                
            if hasattr(block, 'conv2') and block.conv2.kernel_size == (3, 3):
                new_conv = DeformableBlock(
                    block.conv2.in_channels, 
                    block.conv2.out_channels, 
                    kernel_size=3, 
                    stride=block.conv2.stride, 
                    padding=block.conv2.padding, 
                    bias=False
                )
                new_conv.dcn.weight.data = block.conv2.weight.data
                block.conv2 = new_conv

# --- Model Factory ---

# import timm (Moved to local scope to avoid startup errors if missing)

def get_available_models():
    return {
        'resnet18': 'ResNet18 (Standard)',
        'resnet50': 'ResNet50 (Deep)',
        'efficientnet_b0': 'EfficientNet-B0 (Efficient)',
        'dcn': 'Deformable CNN (Advanced)',
        'eva02': 'EVA-02 ViT (Transformer)'
    }

def create_model(model_name, num_classes, device):
    print(f"[Model Factory] Initializing {model_name}...", flush=True)
    
    model = None
    
    # 1. Base Model Creation & Configuration
    if model_name == 'dcn':
        # DCN uses ResNet18 as base
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        print("[Model Factory] Applying Deformable Convolutions...", flush=True)
        _replace_layers_with_dcn(model)
        
    elif model_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        
    elif model_name == 'eva02':
        # Using EVA-02 Base Patch14 224
        # Note: Requires timm installed
        print("[Model Factory] Loading EVA-02 from timm...", flush=True)
        import timm
        try:
            model = timm.create_model('eva02_base_patch14_224.mim_in22k_ft_in1k', pretrained=True)
        except Exception:
            # Fallback if specific tag fails or newer timm version
            print("[Model Factory] Specific EVA-02 tag failed, trying generic 'eva02_base_patch14_224'...", flush=True)
            model = timm.create_model('eva02_base_patch14_224', pretrained=True)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 2. Freezing Strategy (Transfer Learning)
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze specific layers based on model type
    if model_name == 'dcn':
        # Unfreeze DCN layers
        for name, module in model.named_modules():
            if isinstance(module, DeformableBlock):
                for param in module.parameters():
                    param.requires_grad = True
    
    # 3. Final Layer Modification (always trainable)
    if model_name == 'eva02':
        # timm helper to reset head to num_classes
        model.reset_classifier(num_classes)
    elif model_name == 'efficientnet_b0':
        # EfficientNet uses a classifier with Dropout
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        # Standard ResNet approach
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    # 4. Move to Device
    model = model.to(device)
    
    # 5. Return model and optimized parameters
    parameters_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    return model, parameters_to_optimize
