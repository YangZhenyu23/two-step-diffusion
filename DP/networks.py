import torch
import torch.nn as nn
import torch.nn.init as init


class MLP(nn.Module):
    def __init__(self, layer_sizes, activation=torch.relu):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])

            # --- 添加初始化代码 ---
            # 使用 Xavier 正态分布初始化权重
            nn.init.xavier_normal_(linear_layer.weight)
            # 偏置通常初始化为 0
            nn.init.zeros_(linear_layer.bias)
            # ----------------------

            self.layers.append(linear_layer)

        self.activation = activation

    def forward(self, x):
        # 遍历除最后一层外的所有层，并应用激活函数
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        # 最后一层通常不加激活函数（或者根据需求另加）
        return self.layers[-1](x)


import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=torch.tanh):
        super(ResidualBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation

        # 如果输入输出维度不一致，定义一个映射层来对齐维度
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        residual = self.shortcut(x)  # 处理残差分支
        out = self.linear(x)  # 处理主路径
        out = self.activation(out)

        return out + residual  # 核心：残差相加


class ResMLP(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh):
        super(ResMLP, self).__init__()

        self.layers = nn.ModuleList()
        # 构建残差层（不包括最后一层）
        for i in range(len(layer_sizes) - 2):
            self.layers.append(ResidualBlock(layer_sizes[i], layer_sizes[i + 1], activation))

        # 最后一层通常保持普通线性层，不加残差
        self.output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self._initialize_weights()

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)