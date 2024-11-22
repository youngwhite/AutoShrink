import torch.nn as nn

def get_model(
    model_name:str,
    pretrain: bool=True,
    num_classes: int=1000
    ):
    if model_name == 'resnet50':
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrain else None)        
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrain else None)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    elif model_name == 'efficientnet_v2_s':
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrain else None)
        model.classifier[-1] = nn.Linear(1280, num_classes)
    elif model_name == 'vit_b_16':
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrain else None)
        model.heads.head = nn.Linear(768, num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

def get_wrapped_model(
    model_name:str,
    pretrain: bool=True,
    num_classes: int=10
    ):
    if model_name == 'mobilenet_v2':
        from .mobilenet_v2 import MNv2
        model = MNv2(pretrain=pretrain, num_classes=num_classes)
    elif model_name == 'vit_b_16':
        from .vit_b_16 import ViTb_16
        model = ViTb_16(pretrain=pretrain, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

if __name__ == '__main__':
    import torch
    X = torch.rand(1, 3, 224, 224)
    # test models
    for name in ['mobilenet_v2', 'vit_b_16']:
        model = get_model(name, num_classes=10)
        print(f"Model name: {name}, output length: {model(X).shape}")
    
    # test wrapped models
    for name in ['mobilenet_v2', 'vit_b_16']:
        model = get_wrapped_model(name, num_classes=100)
        model = model.retain_layers(1)
        print(f"Model name: {name}, output length: {len(model(X))}")

    # -m Python 会将它视作模块的一部分，而不是独立脚本。
    # python -m models.model_getter