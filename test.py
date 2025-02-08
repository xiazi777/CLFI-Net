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

def main():
    parser = argparse.ArgumentParser('CLFI-Net', add_help=False)
    parser.add_argument('--batch_size', type=int, default=8, help="batch size for training")
    parser.add_argument('--dataset_name', type=str, default="air", help="dataset name")
    parser.add_argument('--topn', type=int, default=4, help="parts number")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone")
    parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
    args, _ = parser.parse_known_args()
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
    elif dataset_name == 'algae':
        trainset = ALGAE(root=data_root, is_train=True, data_len=None)
        testset = ALGAE(root=data_root, is_train=False, data_len=None)
    elif dataset_name == 'cub':
        trainset = CUB(root=data_root, is_train=True, data_len=None)
        testset = CUB(root=data_root, is_train=False, data_len=None)

    ## Output
    topn = args.topn

    ## Model
    net = load_model(backbone=args.backbone, pretrain=True, require_grad=True, classes_num=classes_num, topn=topn)
    model_weight_path = "./air_resnet50_4/model.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    store_dict = torch.load(model_weight_path)
    net.load_state_dict(store_dict)
    print(net)
    net.cuda()

    acc1, acc2, acc3, acc4, acc_test = test(net, testset, batch_size)
    result_str = ' acc1 = %.5f | acc2 = %.5f | acc3 = %.5f | acc4 = %.5f | acc_test = %.5f \n' % (acc1, acc2, acc3, acc4, acc_test)
    print(result_str)



if __name__ == "__main__":
    main()
