# codes adapted from https://github.com/TsingZ0/PFL-Non-IID/blob/70d1a8f1a372eaabc3305a4566d15246031c8b8c/dataset/generate_cifar10.py

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils import check, separate_data, split_data, save_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default = 20,
                    help='number of clients')
parser.add_argument('--dir', type=str, help='directory for storing the data',
                    default ="/mlodata1/dongyang/Cifar10/" )
parser.add_argument('--niid',  action='store_true', help='sample non-iid data for each worker' )
parser.add_argument('--balance',action='store_true')
parser.add_argument('--partition',type=str,default='dir' )
parser.add_argument('--alpha',type=float ,default=0.1 ,help='needed when using dirichelet distr')
parser.add_argument('--refgen',action='store_true')


random.seed(1)
np.random.seed(1)
num_classes = 10



# Allocate data to users
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition,alpha,ref):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    ref_path = dir_path + 'ref/'

    if check(config_path, train_path, test_path, ref_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=dir_path+"rawdata", train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    niid, balance, partition)
    train_data, test_data, ref_data = split_data(X, y,ref)
    save_file(config_path, train_path, test_path, train_data, test_data, ref_data, num_clients, num_classes, 
        statistic, alpha, ref_path, niid, balance, partition)


if __name__ == "__main__":
    args = parser.parse_args()
    num_clients = args.n_clients
    dir_path = args.dir
    niid = args.niid
    balance = args.balance
    partition = args.partition
    alpha = args.alpha
    ref = args.refgen
    print('num_clients:',num_clients,'\n')
    print('niid:',niid,'\n')
    print('partition:',partition,'\n')
    if partition == 'dir':
        print('alpha:',alpha,'\n')
    print('generate refdata?:',ref,'\n')
    #niid = True if sys.argv[1] == "noniid" else False
    #balance = True if sys.argv[2] == "balance" else False
    #partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition,alpha,ref)
