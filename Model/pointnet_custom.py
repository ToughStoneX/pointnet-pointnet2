# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 10:48
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : pointnet_custom.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchsummary import summary
import os

def conv_bn_block(input, output, kernel_size):
    '''
    标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    '''
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )

class PointNet_Custom1(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet_Custom1, self).__init__()

        self.num_classes = num_classes

        # block 1
        # self.conv_1 = nn.Conv1d(3, 32, 1)
        # self.bn_1 = nn.BatchNorm1d(32)
        # self.relu_1 = nn.ReLU(inplace=True)
        self.block_1 = conv_bn_block(3, 32, 1)

        # block 2
        # self.conv_2 = nn.Conv1d(32, 32, 1)
        # self.bn_2 = nn.BatchNorm1d(32)
        # self.relu_2 = nn.ReLU(inplace=True)
        self.block_2 = conv_bn_block(32, 32, 1)

        # block 3
        # self.conv_3 = nn.Conv1d(64, 64, 1)
        # self.bn_3 = nn.BatchNorm1d(64)
        # self.relu_3 = nn.ReLU(inplace=True)
        self.block_3 = conv_bn_block(64, 64, 1)

        # block 4
        # self.conv_4 = nn.Conv1d(64, 64, 1)
        # self.bn_4 = nn.BatchNorm1d(64)
        # self.relu_4 = nn.ReLU(inplace=True)
        self.block_4 = conv_bn_block(64, 64, 1)

        # block 5
        self.block_5 = conv_bn_block(128, 128, 1)
        # block 6
        self.block_6 = conv_bn_block(128, 128, 1)

        # block 7
        self.block_7 = conv_bn_block(256, 256, 1)
        # block 8
        self.block_8 = conv_bn_block(256, 256, 1)

        # block 9
        self.block_9 = conv_bn_block(512, 512, 1)

        self.fc_10 = nn.Linear(512, 512)
        self.drop_10 = nn.Dropout(0.5)
        self.fc_11 = nn.Linear(512, 256)
        self.drop_11 = nn.Dropout(0.5)
        self.fc_12 = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N = int(x.shape[0]), int(x.shape[2])

        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x = torch.cat([x1, x2], 1)

        x1 = self.block_3(x)
        x2 = self.block_4(x1)
        x = torch.cat([x1, x2], 1)

        x1 = self.block_5(x)
        x2 = self.block_6(x1)
        x = torch.cat([x1, x2], 1)

        x1 = self.block_7(x)
        x2 = self.block_8(x1)
        x = torch.cat([x1, x2], 1)

        x = self.block_9(x)

        x = nn.MaxPool1d(N)(x)
        x = x.view(B, 512)

        x = self.fc_10(x)
        x = self.drop_10(x)

        x = self.fc_11(x)
        x = self.drop_11(x)

        x = self.fc_12(x)
        x = F.log_softmax(x, dim=-1)

        return x

class PointNet_Custom2(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet_Custom2, self).__init__()

        self.num_classes = num_classes

        # block 1
        self.block_1 = conv_bn_block(3, 32, 1)
        # block 2
        self.block_2 = conv_bn_block(32, 32, 1)

        # block 3
        self.block_3 = conv_bn_block(64, 64, 1)
        # block 4
        self.block_4 = conv_bn_block(64, 64, 1)

        # block 5
        self.block_5 = conv_bn_block(128, 128, 1)
        # block 6
        self.block_6 = conv_bn_block(128, 128, 1)

        # block 7
        self.block_7 = conv_bn_block(256, 256, 1)
        # block 8
        self.block_8 = conv_bn_block(256, 256, 1)

        # block 9
        self.block_9 = conv_bn_block(512, 512, 1)

        self.fc_10 = nn.Linear(960, 512)
        self.drop_10 = nn.Dropout(0.5)
        self.fc_11 = nn.Linear(512, 256)
        self.drop_11 = nn.Dropout(0.5)
        self.fc_12 = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N = int(x.shape[0]), int(x.shape[2])

        x1 = self.block_1(x)
        x2 = self.block_2(x1)
        x = torch.cat([x1, x2], 1)

        x1 = self.block_3(x)
        f1 = nn.MaxPool1d(N)(x1)
        f1 = f1.view(B, 64)
        x2 = self.block_4(x1)
        x = torch.cat([x1, x2], 1)

        x1 = self.block_5(x)
        f2 = nn.MaxPool1d(N)(x1)
        f2 = f2.view(B, 128)
        x2 = self.block_6(x1)
        x = torch.cat([x1, x2], 1)

        x1 = self.block_7(x)
        f3 = nn.MaxPool1d(N)(x1)
        f3 = f3.view(B, 256)
        x2 = self.block_8(x1)
        x = torch.cat([x1, x2], 1)

        x = self.block_9(x)

        x = nn.MaxPool1d(N)(x)
        x = x.view(B, 512)
        x = torch.cat([f1, f2, f3, x], dim=1)

        x = self.fc_10(x)
        x = self.drop_10(x)

        x = self.fc_11(x)
        x = self.drop_11(x)

        x = self.fc_12(x)
        x = F.log_softmax(x, dim=-1)

        return x


if __name__ == '__main__':
    # net = PointNet_Custom1()
    net = PointNet_Custom2()

    summary(net, (3, 2048))


