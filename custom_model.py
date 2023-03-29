'''
File: custom_model.py
Project: fid-scoring-mri
File Created: 2023-03-27 20:07:56
Author: sangminlee
-----
This script ...
Reference
...
'''
import torch
import timm


class CustomNet(torch.nn.Module):
    def __init__(self, target_layer_idx):
        super(CustomNet, self).__init__()
        m = timm.create_model('inception_v3', pretrained=True, num_classes=0)
        self.m_before = torch.nn.Sequential(*list(m.children())[:target_layer_idx])
        self.m_after = torch.nn.Sequential(*list(m.children())[target_layer_idx:])

    def before_forward(self, x):
        return self.m_before(x)

    def after_forward(self, x):
        return self.m_after(x)

    def forward(self, x):
        self.feature = self.m_before(x)
        self.output = self.m_after(self.feature)
        return self.output, self.feature
