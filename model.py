import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from anchors import generate_anchor, hard_nms


class CLFI(nn.Module):
    def __init__(self, model, feature_size, num_ftrs, classes_num, topn):
        super(CLFI, self).__init__()

        self.backbone = model
        self.num_ftrs = num_ftrs
        self.topn = topn
        self.im_sz = 448
        self.pad_side = 224

        self.epn = Enhance_ProposalNet(self.num_ftrs)
        _, edge_anchors, _ = generate_anchor()
        self.edge_anchors = (edge_anchors + self.pad_side).astype(int)

        self.GMP = nn.AdaptiveMaxPool2d(1)
        self.gl_mlp1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 4),
            nn.Linear(self.num_ftrs // 4, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        self.gl_mlp2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        self.gl_mlp3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )

        self.conv_block1 = nn.Sequential(
            CA(self.num_ftrs // 4, self.num_ftrs // 2, 16),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        self.conv_block2 = nn.Sequential(
            CA(self.num_ftrs // 2, self.num_ftrs // 2, 16),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        self.conv_block3 = nn.Sequential(
            CA(self.num_ftrs, self.num_ftrs // 2, 16),
            nn.AdaptiveMaxPool2d(1)
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2),
            nn.Linear(self.num_ftrs // 2, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_size, classes_num),
        )
        self.classifier_concat = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs // 2 * 3),
            nn.Linear(self.num_ftrs // 2 * 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, classes_num),
        )
        self.conv_down1 = nn.Conv2d(self.num_ftrs // 4, self.num_ftrs // 4, kernel_size = 3, stride = 2, padding = 1)
        self.conv_down2 = nn.Conv2d(self.num_ftrs // 2, self.num_ftrs // 2, kernel_size = 3, stride = 2, padding = 1)
        self.conv_down3 = nn.Conv2d(self.num_ftrs, self.num_ftrs, kernel_size = 3, stride = 2, padding = 1)
        self.glinteraction = GLinteraction(self.topn)

    def forward(self, x, is_train=True):
        _, _, f1, f2, f3 = self.backbone(x)

        batch = x.shape[0]
        rpn_score = self.epn(f3.detach())
        all_cdds = [np.concatenate((x.reshape(-1, 1),
                                    self.edge_anchors.copy(),
                                    np.arange(0, len(x)).reshape(-1, 1)),
                                   axis=1) for x in rpn_score.data.cpu().numpy()]
        top_n_cdds = np.array([hard_nms(x, self.topn, iou_thresh=0.25) for x in all_cdds])
        top_n_index = top_n_cdds[:, :, -1].astype(int)
        top_n_index = torch.from_numpy(top_n_index).long().to(x.device)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)
        part_imgs = torch.zeros([batch, self.topn, 3, 224, 224]).to(
            x.device)

        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant',
                      value=0)
        for i in range(batch):
            for j in range(self.topn):
                [y0, x0, y1, x1] = top_n_cdds[i, j, 1:5].astype(int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1],
                                                      size=(224, 224), mode='bilinear',
                                                      align_corners=True)

        part_imgs = part_imgs.view(batch * self.topn, 3, 224, 224)
        _, _, f1_part, f2_part, f3_part = self.backbone(
            part_imgs.detach())

        f1_g = self.conv_down1(f1)
        f2_g = self.conv_down2(f2)
        f3_g = self.conv_down3(f3)
        f1_gl = self.glinteraction(f1_g, f1_part)
        f2_gl = self.glinteraction(f2_g, f2_part)
        f3_gl = self.glinteraction(f3_g, f3_part)
        f1_gl = self.GMP(f1_gl).view(batch, -1)
        f2_gl = self.GMP(f2_gl).view(batch, -1)
        f3_gl = self.GMP(f3_gl).view(batch, -1)
        ygl1 = self.gl_mlp1(f1_gl)
        ygl2 = self.gl_mlp2(f2_gl)
        ygl3 = self.gl_mlp3(f3_gl)

        f1_part = self.conv_block1(f1_part).view(batch * self.topn, -1)
        f2_part = self.conv_block2(f2_part).view(batch * self.topn, -1)
        f3_part = self.conv_block3(f3_part).view(batch * self.topn, -1)
        yp1 = self.classifier1(f1_part)
        yp2 = self.classifier2(f2_part)
        yp3 = self.classifier3(f3_part)
        yp4 = self.classifier_concat(torch.cat((f1_part, f2_part, f3_part), -1))

        f1 = self.conv_block1(f1).view(batch, -1)
        f2 = self.conv_block2(f2).view(batch, -1)
        f3 = self.conv_block3(f3).view(batch, -1)
        y1 = self.classifier1(f1)
        y2 = self.classifier2(f2)
        y3 = self.classifier3(f3)
        y4 = self.classifier_concat(torch.cat((f1, f2, f3), -1))

        return f3,y1, y2, y3, y4, yp1, yp2, yp3, yp4, top_n_prob, ygl1, ygl2, ygl3,top_n_cdds


class GLinteraction(nn.Module):
    def __init__(self, topn):
        super(GLinteraction, self).__init__()
        self.topn = topn
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, parts):
        B, C, H, W = x.shape
        x_reshape = x.view(B, C, -1)
        parts = parts.view(B, self.topn, C, H, W)
        parts_cat = parts.permute(0, 2, 1, 3, 4).reshape(B, self.topn * C, H, W)
        parts_cat = parts_cat.view(B, self.topn * C, -1)
        parts_cat_T = parts_cat.permute(0, 2, 1)
        weights = torch.matmul(x_reshape, parts_cat_T)
        weights = self.softmax(weights)
        x_gl = torch.matmul(weights, parts_cat)
        x_gl = x_gl.view(B, C, H, W)
        x_gl = x_gl + x

        return x_gl


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CA(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(CA, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                               bias=False)
        self.conv3 = nn.Conv2d(in_planes, in_planes, kernel_size=5, stride=1, padding=2, dilation=1, groups=1,
                               bias=False)
        self.conv_cat = nn.Conv2d(in_planes * 3, in_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                                  bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                              bias=False)

    def forward(self, x):
        x_o = x
        x_1x1 = self.conv1(x)
        x_3x3 = self.conv2(x)
        x_5x5 = self.conv3(x)
        x_xall = torch.cat((x_1x1, x_3x3, x_5x5), dim=1)
        x = self.conv_cat(x_xall)
        x = x + x_o
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x_o
        y = self.conv(y)
        return y


class Enhance_ProposalNet(nn.Module):
    def __init__(self, depth):
        super(Enhance_ProposalNet, self).__init__()
        self.feature_enhance = nn.Sequential(
            BasicConv(depth, depth, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(depth, depth, kernel_size=5, stride=1, padding=2, relu=True),
            BasicConv(depth, depth, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(depth, depth * 4, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(depth * 4, depth, kernel_size=1, stride=1, padding=0, relu=True)
        )
        self.down1 = nn.Conv2d(depth, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)
        # proposals: 14x14x6, 7x7x6, 4x4x9

    def forward(self, x):
        batch_size = x.size(0)  # x=f3[b,14,14,2048]
        x = self.feature_enhance(x)
        d1 = self.ReLU(self.down1(x))  # d1[b,14,14,128]
        d2 = self.ReLU(self.down2(d1))  # d2[b,7,7,128]
        d3 = self.ReLU(self.down3(d2))  # d3[b,4,4,128]
        t1 = self.tidy1(d1).view(batch_size, -1)  # [b,14,14,6]->[b,14*14*6]
        t2 = self.tidy2(d2).view(batch_size, -1)  # [b,7,7,6]->[b,7*7*6]
        t3 = self.tidy3(d3).view(batch_size, -1)  # [b,4,4,9]->[b,4*4*9]
        return torch.cat((t1, t2, t3), dim=1)  # [b,14*14*6+7*7*6+4*4*9]


