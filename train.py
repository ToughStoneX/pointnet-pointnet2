# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 9:03
# @Author  : Xu HongBin
# @Email   : 2775751197@qq.com
# @github  : https://github.com/ToughStoneX
# @blog    : https://blog.csdn.net/hongbin_xu
# @File    : train.py
# @Software: PyCharm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import logging  # 引入logging模块
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import warnings
warnings.filterwarnings("ignore")


from dataset import ModelNet40DataSet
from Model.pointnet import PointNet, PointNet_Vanilla
from Model.pointnet2 import PointNet2ClsMsg
from Model.pointnet_custom import PointNet_Custom1, PointNet_Custom2

batch_size = 32
num_epochs = 50
test_freq = 5
save_freq = 5
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train():
    pwd = os.getcwd()
    weights_dir = os.path.join(pwd, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    logging.info('Loading Dataset...')
    train_dataset = ModelNet40DataSet(train=True)
    test_dataset = ModelNet40DataSet(train=False)
    logging.info('train_dataset: {}'.format(len(train_dataset)))
    logging.info('test_dataset: {}'.format(len(test_dataset)))
    logging.info('Done...\n')


    logging.info('Creating DataLoader...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    logging.info('Done...\n')


    logging.info('Checking gpu...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logging.info('gpu available: {}'.format(torch.cuda.device_count()))
        logging.info('current gpu: {}'.format(torch.cuda.get_device_name(0)))
        logging.info('gpu capability: {}'.format(torch.cuda.get_device_capability(0)))
    else:
        logging.info('gpu not available, running on cpu instead.')
    logging.info('Done...\n')


    logging.info('Create SummaryWriter in ./summary')
    # 创建SummaryWriter
    summary_writer = SummaryWriter(comment='PointNet', log_dir='summary')
    logging.info('Done...\n')


    logging.info('Creating Model...')
    # create pointnet model
    # model = PointNet(num_classes=40).to(device)
    # create pointnet(vanilla) model
    # model = PointNet_Vanilla(num_classes=40).to(device)
    # create pointnet++ model
    # model = PointNet2ClsMsg(num_classes=40).to(device)
    # custom pointnet vanilla
    # model = PointNet_Custom1(num_classes=40).to(device)
    # custom pointnet vanilla 2
    model = PointNet_Custom2(num_classes=40).to(device)
    # add graph
    # dummy_input = torch.rand(10, 3, 2048).to(device)
    # summary_writer.add_graph(model, dummy_input)
    # CrossEntropy Loss
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters())
    logging.info('Done...\n')


    logging.info('Start training...')
    for epoch in range(1, num_epochs+1):
        logging.info("--------Epoch {}--------".format(epoch))

        tqdm_batch = tqdm(train_loader, desc='Epoch-{} training'.format(epoch))

        # train
        model.train()
        loss_tracker = AverageMeter()
        for batch_idx, (data, label) in enumerate(tqdm_batch):
            data, label = data.to(device), label.to(device)
            # print(data.size())
            data = data.permute(0, 2, 1)

            out = model(data)

            # print('out: {}, label: {}'.format(out.size(), label.size()))
            loss = criterion(out, label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_tracker.update(loss.item(), label.size(0))

            del data, label

        tqdm_batch.close()
        logging.info('Loss: {:.4f} ({:.4f})'.format(loss_tracker.val, loss_tracker.avg))

        summary_writer.add_scalar('loss', loss_tracker.avg, epoch)

        if epoch % test_freq == 0:
            tqdm_batch = tqdm(test_loader, desc='Epoch-{} testing'.format(epoch))

            model.eval()
            correct_cnt = 0
            total_cnt = 0
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(tqdm_batch):
                    data, label = data.to(device), label.to(device)
                    data = data.permute(0, 2, 1)

                    out = model(data)
                    pred_choice = out.max(1)[1]

                    correct_cnt += pred_choice.eq(label.view(-1)).sum().item()
                    total_cnt += label.size(0)

                    del data, label

            acc = correct_cnt / total_cnt
            logging.info('Accuracy: {:.4f}'.format(acc))

            summary_writer.add_scalar('acc', acc, epoch)

            tqdm_batch.close()

        if epoch % save_freq == 0:
            ckpt_name = os.path.join(weights_dir, 'pointnet_{0}.pth'.format(epoch))
            torch.save(model.state_dict(), ckpt_name)
            logging.info('model saved in {}'.format(ckpt_name))

    summary_writer.close()


if __name__ == '__main__':
    train()