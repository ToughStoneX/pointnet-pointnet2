# -*- coding: utf-8 -*-
# @Time    : 2018/11/10 2:28
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : pointnet2.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from torchsummary import summary

'''
# 函数说明：
#     计算两个点集中所有点之间的欧氏距离
# 注释：
# 以B=1，C=3的情况为例：
# src的维度为 [N, 3]，dst的维度为 [M, 3]。
# 直观上理解为src有 N个点， dst有 M个点，返回的就是src中N个点与dst中的M个点两两之间的欧式距离，所以返回的距离应该是一个 [N, M]的矩阵。
# 由于src和dst中可能不止一个点集（样本），所以扩展到B维。最后返回的结果应该是一个 [B, N, M]的矩阵。
# 假设src中的坐标表示为 [xn, yn, zn]，dst中的坐标表示为 [xm, ym, zm]。
# 那么，src^T * dst = xn*xm + yn*ym + zn*zm；
# sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
# sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
# dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#      = sum(src**2, dim=-1) + sum(dst**2, dim=-1) - 2*src^T * dst
# 使用这种方法求解欧式距离，然后增加一个维度B即可。
'''


def square_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, D1, D2, ..., Dn]
    Return:
        new_points:, indexed points data, [B, D1, D2, ..., Dn, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


'''
# 函数说明：
#         求解最远的样本，FPS算法实现
# 注释：
#    B表示有多少个样本，N表示每个样本的点数，C为坐标数（通常是xyz坐标），S表示想采样的样本数
#         先随机初始化一个centroids矩阵，大小为B*S；
#    distance矩阵初始化为B*N矩阵，初值给个比较大的值，后面迭代更新；
#    farthest表示最远的点，也是随机初始化，范围为0-N，初始化B个，对应到每个样本都随机有一个初始最远点
#    batch_indices初始化为0-(B-1)的数组
#        考虑特殊的情况，令B为1，C为3：
#        只有一个样本，样本维度有N个点，每个点坐标维度为3。
#        那么，centroids有S个中心点，distance有N个距离值，farthest是一个随机初始化的0-N的一个索引值，batch_indices为0，可以先不管。
#        总共要采样S个sample，所以要遍历S次，i为0-(S-1)的索引值：
#        先将当前的farthest赋值给第i个centroids；
#        取出这个farthest对应的中心点坐标centroid；
#        求出所有点到这个farthest点的欧式距离，存在dist矩阵中；
#        建立一个mask，如果dist中的元素小于distance矩阵中保存的距离值，则更新distance中的对应值，且mask的对应位置1；
#        最后从distance矩阵取出最远的点，用于下一轮迭代
'''


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud data, [B, npoint, C]
    """
    device = xyz.device
    B, N, C = xyz.shape
    S = npoint
    centroids = torch.zeros(B, S, dtype=torch.long).to(device)
    distances = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(S):
        # 更新第i个最远点
        centroids[:, i] = farthest
        # 取出这个最远点的xyz坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        # 计算点集中的所有点到这个最远点的欧式距离
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # 找到距离小于这个点的最大距离的，更新distances
        mask = dist < distances
        distances[mask] = dist[mask]
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        farthest = torch.max(distances, -1)[1]
    return centroids


'''
函数说明：
    寻找球形领域中的点
'''


def query_ball_points(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    K = nsample
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    # new_xyz: [B, S, C], xyz: [B, N, C]
    # sqrdists: [B, S, N]
    sqrdists = square_distance(new_xyz, xyz)
    # 找到所有距离大于radius^2的，其group_idx直接置为N；其余的保留原来的值
    group_idx[sqrdists > radius ** 2] = N
    # 做升序排列，前面大于radius^2的都是N，会是最大值，所以会直接在剩下的点中取出前k个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :K]
    # 考虑到有可能前k个点钟也有被赋值为N的点，这种点需要舍弃，直接用第一个点来代替即可
    # group_first: [B, S, k]， 实际就是把group_idx中的第一个点的值复制为了[B, S, K]的维度
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, K])
    # 找到group_idx中值等于N的点
    mask = group_idx == N
    # 将这些点的值替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]  ? [B, S, C]
        new_points: sampled points data, [B, 1, N, C+D]   ? [B, S, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint

    # 从 [B, N, C]的xyz中根据[B, S, C]的idx(farthest_point_sample(xyz, npoint))取出new_xyz: [B, S, C]
    new_xyz = index_points(xyz, farthest_point_sample(xyz, npoint))
    # idx: [B, S, K]
    idx = query_ball_points(radius, nsample, xyz, new_xyz)
    # grouped_xyz: [B, S, K, C]
    grouped_xyz = index_points(xyz, idx)
    # grouped_xyz减去中心值
    grouped_xyz -= new_xyz.view(B, S, 1, C)
    # 如果有额外的维度，points: [B, N, D]
    if points is not None:
        # grouped_points: [B, S, K, D]
        grouped_points = index_points(points, idx)
        # new_points: [B, S, K, C+D]
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        # new_points: [B, S, K, C]
        new_points = grouped_xyz
    return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    # new_xyz: [B, 1, C]
    new_xyz = torch.zeros(B, 1, C).to(device)
    # grouped_xyz: [B, 1, N, C]
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        # grouped_xyz: [B, 1, N, C]
        # points.view(B, 1, N, -1): [B, 1, N, D]
        # new_points: [B, 1, N, (C+D)]
        #         new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
        new_points = points.view(B, 1, N, -1)
    else:
        # new_points: [B, 1, N, C]
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #         print(points.shape)
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            # new_xyz: [B, 1, C]
            # new_points: [B, 1, N, C+D]
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            # new_xyz: [B, S, C]
            # new_points: [B, S, K, C+D]
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)

        #         print(new_points.shape)
        # new_points: [B, C+D, K, S]
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # 替代了MaxPooling操作
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # print('xyz: {}'.format(xyz.shape))
        # xyz: [B, C, N]
        xyz = xyz.permute(0, 2, 1)
        # xyz: [B, N, C]
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_points(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1: [B, C, N]
        xyz1 = xyz1.permute(0, 2, 1)
        # xyz1: [B, N, C]
        # xyz2: [B, C, S]
        xyz2 = xyz2.permute(0, 2, 1)
        # xyz2: [B, S, C]
        # points2: [B, D, S]
        points2 = points2.permute(0, 2, 1)
        # points2: [B, S, D]
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        #         print(xyz1.shape)           # [8, 64, 3]
        #         print(xyz2.shape)           # [8, 16, 3]
        #         print(points1.shape)        # [8, 256, 64]
        #         print(points2.shape)        # [8, 16, 512]

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # xyz1: [B, N, C]
            # xyz2: [B, S, C]
            #             print(xyz1.shape)
            #             print(xyz2.shape)
            dists = square_distance(xyz1, xyz2)
            # dists: [B, N, S]
            #             print(dists.shape)      # [8, 64, 16]
            dists, idx = dists.sort(dim=-1)
            # dists: [B, N, 3]
            # idx: [B, N, 3]
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            #             print(dists.shape)      # [8, 64, 3]
            #             print(idx.shape)        # [8, 64, 3]
            dists[dists < 1e-10] = 1e-10
            # weight: [B, N, 3]
            weight = 1 / dists
            weight = weight / torch.sum(weight, dim=-1).view(B, N, 1)
            # index_points(points2, idx): [B, N, 3]
            #             print(index_points(points2, idx).shape)        # [8, 64, 3, 512]
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)
        #             print(interpolated_points.shape)        # [8, 64, 512]

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        #         print(new_points.shape)        # [8, 64, 768]

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class PointNet2ClsMsg(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet2ClsMsg, self).__init__()
        self.num_classes = num_classes
        self.sa1 = PointNetSetAbstractionMsg(512,  # npoint
                                             [0.1, 0.2, 0.4],  # radius_list
                                             [16, 32, 128],  # nsample_list
                                             0,  # in_channel
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]]  # mlp_list
                                             )

        self.sa2 = PointNetSetAbstractionMsg(128,  # npoint
                                             [0.2, 0.4, 0.8],  # radius_list
                                             [32, 64, 128],  # nsample_list
                                             320,  # in_channel
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]]  # mlp_list
                                             )

        # npoint, radius, nsample, in_channel, mlp, mlp2, group_all
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None,
                                          in_channel=640, mlp=[256, 512, 1024], mlp2=None, group_all=True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        # modelnet40共有40个类
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, None)
        # print(l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        # print(l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(self.bn1(self.fc1(x)))
        x = self.drop2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        return x


if __name__ == '__main__':
    net = PointNet2ClsMsg()

    summary(net, (3, 2048))

    # xyz = torch.rand(8, 3, 1024)
    # out = net(xyz)
    # print('out: {}'.format(out.shape))