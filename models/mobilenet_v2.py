import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MNv2(nn.Module):
    def __init__(self, pretrain=True, num_classes=10):
        super().__init__()
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1 if pretrain else None)
        self.features = model.features        
        self.classifiers = nn.ModuleList()
        self.num_classes = num_classes

        # 为每个 block 添加分类头
        x = torch.rand(1, 3, 224, 224)
        for block in self.features:
            x = block(x)
            num_features = x.flatten(start_dim=1).shape[-1]
            self.classifiers.append(self._make_classifier(num_features, num_classes))

        # 保留最后一层的原始分类器
        if num_classes == 1000:
            self.classifiers[-1] = model.classifier

    def _make_classifier(self, num_features, num_classes):
        return nn.Sequential(
            nn.Flatten(start_dim=1), 
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        outputs = []
        for idx, (block, classifier) in enumerate(zip(self.features, self.classifiers)):
            x = block(x)
            output = classifier(x)
            outputs.append(output)
        return outputs

    def retain_layers(self, bN: int):
        """返回特定层的特征提取器和分类器"""
        return nn.Sequential(
            self.features[:bN+1],
            self.classifiers[bN]
        )

if __name__ == '__main__':

    x = torch.randn(2, 3, 224, 224)

    model = MNv2(pretrain=True, num_classes=10)
    print('length:', len(model(x)))

    pruned_model = model.retain_layers(14)
    print('output.shape:', pruned_model(x).shape)
