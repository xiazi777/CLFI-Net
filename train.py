from __future__ import print_function
from distutils.log import error
import os
from numpy import result_type
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
from dataset import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

def train():
    parser = argparse.ArgumentParser('CLFI', add_help=False)
    parser.add_argument('--epochs', type=int, default=120, help="training epochs")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size for training")
    parser.add_argument('--resume', type=str, default="", help="saved model path")
    parser.add_argument('--dataset_name', type=str, default="car", help="dataset name")
    parser.add_argument('--topn', type=int, default=4, help="parts number")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    args, _ = parser.parse_known_args()
    epochs = args.epochs
    batch_size = args.batch_size

    ## Data
    data_config = {"air": [100, "../dataset/fgvc-aircraft-2013b"],
                   "car": [196, "../dataset/stanford_cars"],
                   "cub": [200, "../dataset/CUB_200_2011_official/CUB_200_2011"],
                   "algae": [32, "../dataset/ALGAE"],
                   }
    dataset_name = args.dataset_name
    classes_num, data_root = data_config[dataset_name]
    if dataset_name == 'air':
        trainset = AIR(root=data_root, is_train=True, data_len=None)
        testset = AIR(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'car':
        trainset = CAR(root=data_root, is_train=True, data_len=None)
        testset = CAR(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'cub':
        trainset = CUB(root=data_root, is_train=True, data_len=None)
        testset = CUB(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'algae':
        trainset = ALGAE(root=data_root, is_train=True, data_len=None)
        testset = ALGAE(root=data_root, is_train=False, data_len=None)

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic =True
        torch.backends.cudnn.benchmark =False
    set_seed(42)

    num_workers = 16 if torch.cuda.is_available() else 0
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              drop_last=False)
    topn = args.topn
    exp_dir = dataset_name + '_' + args.backbone + '_' + str(topn)
    os.makedirs(exp_dir, exist_ok=True)

    ## Model
    if args.resume != "":
        net = load_model(backbone=args.backbone, pretrain=True, require_grad=True, classes_num=classes_num, topn=topn)
        store_dict = torch.load(args.resume)
        net.load_state_dict(store_dict)
    else:
        net = load_model(backbone=args.backbone, pretrain=True, require_grad=True, classes_num=classes_num, topn=topn)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('cuda')
        net = net.to(device)
        netp = torch.nn.DataParallel(net)
    else:
        device = torch.device('cpu')
        print('cpu')
        netp = net
    print(net)

    ## Train
    CELoss = nn.CrossEntropyLoss()
    deep_paras = [para for name, para in net.named_parameters() if "backbone" not in name]
    optimizer = optim.SGD(
        [{'params': deep_paras},
         {'params': net.backbone.parameters(), 'lr': args.lr / 10.0}],
        lr=args.lr, momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    for epoch in range(1, epochs + 1):
        print('\nEpoch: %d' % epoch)
        optimizer.param_groups[0]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr)
        optimizer.param_groups[1]['lr'] = cosine_anneal_schedule(epoch, epochs, args.lr / 10.0)

        net.train()
        num_correct = [0] * 4
        for _, (inputs, targets) in enumerate(trainloader):
            if inputs.shape[0] < batch_size:
                continue
            if torch.cuda.is_available():
                inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            _,y1, y2, y3, y4, yp1, yp2, yp3, yp4, part_probs, ygl1, ygl2, ygl3 ,topn_cdds= netp(inputs)

            loss1 = smooth_CE(y1, targets, 0.7) * 1
            loss2 = smooth_CE(y2, targets, 0.8) * 1
            loss3 = smooth_CE(y3, targets, 0.9) * 1
            loss4 = smooth_CE(y4, targets, 1) * 1
            loss_img = loss1 + loss2 + loss3 + loss4

            lossgl1 = smooth_CE(ygl1, targets, 0.8)
            lossgl2 = smooth_CE(ygl2, targets, 0.9)
            lossgl3 = smooth_CE(ygl3, targets, 1)
            loss_gl = lossgl1 + lossgl2 + lossgl3

            targets_parts = targets.unsqueeze(1).repeat(1, topn).view(-1)
            lossp1 = smooth_CE(yp1, targets_parts, 0.7)
            lossp2 = smooth_CE(yp2, targets_parts, 0.8)
            lossp3 = smooth_CE(yp3, targets_parts, 0.9)
            lossp4 = smooth_CE(yp4, targets_parts, 1)
            lossp_rank = ranking_loss(part_probs, list_loss(yp4, targets_parts).view(batch_size, topn)) # higher prob, smaller loss
            loss_parts = lossp1 + lossp2 + lossp3 + lossp4 + lossp_rank

            loss = loss_img + loss_parts + 0.1 * loss_gl
            print(loss)

            _, p1 = torch.max(y1.data, 1)
            _, p2 = torch.max(y2.data, 1)
            _, p3 = torch.max(y3.data, 1)
            _, p4 = torch.max(y4.data, 1)

            num_correct[0] += p1.eq(targets.data).cpu().sum()
            num_correct[1] += p2.eq(targets.data).cpu().sum()
            num_correct[2] += p3.eq(targets.data).cpu().sum()
            num_correct[3] += p4.eq(targets.data).cpu().sum()

            loss.backward()
            optimizer.step()

        total = len(trainset)
        print(total)
        acc1 = 100. * float(num_correct[0]) / total
        acc2 = 100. * float(num_correct[1]) / total
        acc3 = 100. * float(num_correct[2]) / total
        acc4 = 100. * float(num_correct[3]) / total

        result_str = 'Iteration %d (train) | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f \n' % (
        epoch, acc1, acc2, acc3, acc4)
        print(result_str)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(result_str)


        if epoch < 5 or epoch % 10 == 0:
            acc1, acc2, acc3, acc4, acc_test = test(net, testset, batch_size)
            if acc_test > max_val_acc:
                max_val_acc = acc_test
                net.cpu()
                torch.save(net.state_dict(), './' + exp_dir + '/model.pth')
                net.to(device)

            result_str = 'Iteration %d | acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_test = %.5f \n' % (
            epoch, acc1, acc2, acc3, acc4, acc_test)
            print(result_str)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write(result_str)


if __name__ == "__main__":
    train()
