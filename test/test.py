import copy
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTb_16(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_token = class_token
        self.features = nn.Sequential(*features)
        self.classifier = classifier
        self.conv_proj = conv_proj
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def preprocess_fn(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        return self.dropout(x)

    def forward(self, x):
        # 确保输入和子模型在同一设备上
        x = self.preprocess_fn(x)
        n = x.shape[0]

        # 增加 class token
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        # 通过特征层并得到最终分类结果
        x = self.features(x)
        return self.classifier(x[:, 0])

def __init__(self, pretrain=True, num_classes=10):
    super().__init__()
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrain else None)

    self.image_size = model.image_size
    self.patch_size = model.patch_size
    self.class_token = model.class_token
    self.hidden_dim = model.hidden_dim

    self.conv_proj = model.conv_proj
    self.class_token = model.class_token
    self.dropout = model.encoder.dropout

    # 初始化 features 和 classifiers
    self.features = nn.ModuleList([block for block in model.encoder.layers])
    self.classifiers = nn.ModuleList([
        nn.Sequential(
            copy.deepcopy(model.encoder.ln),
            nn.Linear(in_features=self.hidden_dim, out_features=num_classes, bias=True)
        )
        for _ in self.features
    ])
    self.cweights = torch.nn.Parameter(torch.ones(len(self.classifiers), requires_grad=True))        

def _process_input(self, x: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    p = self.patch_size
    torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
    torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
    n_h = h // p
    n_w = w // p

    x = self.conv_proj(x)
    x = x.reshape(n, self.hidden_dim, n_h * n_w)
    x = x.permute(0, 2, 1)
    return self.dropout(x)

def forward(self, x: torch.Tensor):
    # 预处理输入
    x = self._process_input(x)
    n = x.shape[0]

    # 增加 class token
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    # 逐层前向传播并获取每层的分类结果
    yy = []
    for block, classifier in zip(self.features, self.classifiers):
        x = block(x)
        yy.append(classifier(x[:, 0]))  # 记录每层的分类结果
    return yy

def retain_layers(self, bN: int):

    sub_model = self._ViTb_16_SubModel(
        class_token=self.class_token,
        features=self.features[:bN + 1],
        classifier=self.classifiers[bN],
        conv_proj=self.conv_proj,
        patch_size=self.patch_size,
        hidden_dim=self.hidden_dim,
        dropout=self.dropout
    )
    return sub_model
