import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
import copy

class ViTb_16(nn.Module):
    def __init__(self, pretrain=True, num_classes=10):
        super().__init__()
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrain else None)
        
        self.patch_size = model.patch_size
        self.image_size = model.image_size
        self.conv_proj = model.conv_proj
        self.hidden_dim = model.hidden_dim
        self._process_input = model._process_input

        self.class_token = model.class_token

        self.features = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        
        for layer in model.encoder.layers:
            self.features.append(layer)
            self.classifiers.append(
                nn.Sequential(
                    copy.deepcopy(model.encoder.ln),
                    nn.Linear(in_features=768, out_features=num_classes, bias=True)
                )
            )

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        yy = []
        for block, classifier in zip(self.features, self.classifiers):
            x = block(x)
            yy.append(classifier(x[:, 0]))
        return yy

    def retain_layers(self, bN: int):
        """
        返回一个裁剪后的新模型实例，仅保留前 bN+1 个特性提取层及对应分类器。
        """
        return RetainedViTb_16(
            features=self.features[:bN+1],
            classifier=self.classifiers[bN],
            process_input=self._process_input,
            class_token=self.class_token,
        )

class RetainedViTb_16(nn.Module):
    def __init__(self, features, classifier, process_input, class_token):
        super().__init__()
        self.features = features
        self.classifier = classifier
        self._process_input = process_input
        self.class_token = class_token

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        n = x.shape[0]

        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        for block in self.features:
            x = block(x)
        return self.classifier(x[:, 0])

if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = ViTb_16(num_classes=10)

    yy = model(x)
    print('length:', len(yy))


    pruned_model = model.retain_layers(5)
    print('output.shape:', pruned_model(x).shape)

    torch.save(pruned_model, 'pruned_model.pth')

    # import torch
    # model = torch.load('/home/usrs/wang.changlong.s8/OPMC/models/pruned_model.pth')
    # x = torch.randn(2, 3, 224, 224)
    # print(model(x).shape)