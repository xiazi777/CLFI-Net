import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
from dataset import *

#构建网络
def load_model(backbone, pretrain=True, require_grad=True, classes_num=200, topn=4):
    print('==> Building model..')
    feature_size = 512
    if backbone == 'resnet50':
        num_ftrs = 2048
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = CLFI(net, feature_size, num_ftrs, classes_num, topn=topn)
    elif backbone == 'resnet101':
        num_ftrs = 2048
        net = resnet101(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = CLFI(net, feature_size, num_ftrs, classes_num)

    return net

def test(net, testset, batch_size):
    
    device = torch.device('cuda')
    num_workers = 16 if torch.cuda.is_available() else 0
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size//2, shuffle=False, num_workers=num_workers, drop_last=False)

    net.eval()
    num_correct = [0] * 5
    for _, (inputs, targets) in enumerate(testloader):
        if torch.cuda.is_available():
            inputs, targets = inputs.to(device), targets.to(device)

        _,y1, y2, y3, y4, _, _, _, _, _, _, _, _ , _= net(inputs, is_train=False)

        _, p1 = torch.max(y1.data, 1)
        _, p2 = torch.max(y2.data, 1)
        _, p3 = torch.max(y3.data, 1)
        _, p4 = torch.max(y4.data, 1)
        _, p5 = torch.max((y1 + y2 + y3 + y4).data, 1)

        num_correct[0] += p1.eq(targets.data).cpu().sum()
        num_correct[1] += p2.eq(targets.data).cpu().sum()
        num_correct[2] += p3.eq(targets.data).cpu().sum()
        num_correct[3] += p4.eq(targets.data).cpu().sum()
        num_correct[4] += p5.eq(targets.data).cpu().sum()
    
    total = len(testset)
    print(total)
    acc1 = 100. * float(num_correct[0]) / total
    acc2 = 100. * float(num_correct[1]) / total
    acc3 = 100. * float(num_correct[2]) / total
    acc4 = 100. * float(num_correct[3]) / total
    acc_test = 100. * float(num_correct[4]) / total

    return acc1, acc2, acc3, acc4, acc_test

def ranking_loss(score, targets):
    if torch.cuda.is_available():
        loss = Variable(torch.zeros(1).cuda())
    else:
        loss = Variable(torch.zeros(1))
    batch_size = score.size(0)

    if torch.cuda.is_available():
        data_type = torch.cuda.FloatTensor
    else:
        data_type = torch.FloatTensor
    for i in range(targets.shape[1]):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(data_type)
        pivot = score[:, i].unsqueeze(1)
        loss_p = (1 - pivot + score) * targets_p
        loss_p = torch.sum(F.relu(loss_p))
        loss += loss_p
    return loss / batch_size


def smooth_CE(logits, label, peak):

    batch, num_cls = logits.shape
    label_logits = np.zeros(logits.shape, dtype=np.float32) + (1-peak)/(num_cls-1)
    ind = ([i for i in range(batch)], list(label.data.cpu().numpy()))
    label_logits[ind] = peak
    smooth_label = torch.from_numpy(label_logits).to(logits.device)

    logits = F.log_softmax(logits, -1)
    ce = torch.mul(logits, smooth_label)
    loss = torch.mean(-torch.sum(ce, -1))

    return loss
