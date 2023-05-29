import torch
from random import sample
import numpy as np
from torch.utils.data import DataLoader
import os
import random

class PDLDataSet(torch.utils.data.Dataset):
    def __init__(self,dataset,mode='normal',num_class=10):
        if mode == 'normal':
            self.X = dataset['x']
            self.y = dataset['y']
        elif mode == 'randomized':
            self.X = dataset['x']
            self.y = random.sample(list(dataset['y']), len(dataset['y']))
        elif mode == 'flipped':
            self.X = dataset['x']
            label_shift = random.randint(1, num_class-1)
            self.y = list((np.array(dataset['y'])+ label_shift)%num_class)


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        img = self.X[idx,:,:,:]
        label = self.y[idx]
        return img, label


def dl_to_sampler(dl):
    dl_iter = iter(dl)
    def sample():
        nonlocal dl_iter
        try:
            return next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            return next(dl_iter)
    return sample

class SharedData(torch.utils.data.Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self,idx):
        img = self.X[idx,:,:,:]
        label = self.Y[idx]
        return img, label,idx


def get_ref_data(curr_path):
    ref_x = []
    ref_y = []
    #ref_client = []
    for dir in os.listdir(curr_path):
        tmp_path = os.path.join(curr_path,dir)
        with open(tmp_path, 'rb') as f:
            tmp = np.load(f, allow_pickle=True)['data'].tolist()
        ref_x.extend(tmp['x'])
        ref_y.extend(tmp['y'])
        #ref_client.extend(dir[:-4]*len(ref_y))
    ref_X = torch.tensor(np.array(ref_x))
    ref_y = torch.tensor(ref_y)
    return ref_X, ref_y

def get_ref_labels_by_client(curr_path):
    ref_labels_by_clients = {}
    for dir in os.listdir(curr_path):
        tmp_path = os.path.join(curr_path,dir)
        client_id = int(dir[:-4])
        with open(tmp_path, 'rb') as f:
            tmp = np.load(f, allow_pickle=True)['data'].tolist()
        ref_labels_by_clients[client_id] =tmp['y']
    return ref_labels_by_clients

def split_data(dataset,ratio=0.8):
    l = len(dataset)
    full_idx = list(np.arange(l))
    train_idx = sample(full_idx,int(0.8*l))
    test_idx = list(set(full_idx)-set(train_idx))
    train_data = [dataset[id] for id in train_idx]
    test_data = [dataset[id] for id in test_idx]
    return train_data, test_data

def get_dataloaders(dataset, ratio = 0.8, train_batch_size=10):
    train_data, test_data = split_data(dataset,ratio=0.8)
    train_loader = get_loader(train_data,batch_size=train_batch_size)
    test_loader = torch.utils.data.DataLoader( test_data, batch_size=len(test_data), shuffle=False)
    return train_loader, test_loader

def get_loader(dataset,batch_size):
    return DataLoader(dataset, batch_size=batch_size)


def save_results(trust_weight_dict,test_accuracy_dict,ref_accuracy_dict, respath, args):
    ref_acc_path = os.path.join(respath,'global_accuracy_'+str(args.dataset_name)+'_'+args.consensus_mode + '_'+args.sim_measure+'_'+args.trust_update+'_'+'lam_'+str(args.lambda_) +'_'+str(args.experiment_no)+'_'+'local_epoch_'+str(args.num_local_epochs)+'_'+args.setting+'.pt')

    test_acc_path = os.path.join(respath,'local_accuracy_'+str(args.dataset_name)+'_'+args.consensus_mode + '_'+args.sim_measure+'_'+args.trust_update+'_'+'lam_'+str(args.lambda_) +'_'+str(args.experiment_no)+'_'+'local_epoch_'+str(args.num_local_epochs)+'_'+args.setting+'.pt')

    trust_weight_path = os.path.join(respath,'trust_weight_dict_'+str(args.dataset_name)+'_'+args.consensus_mode + '_'+args.sim_measure+'_'+args.trust_update+'_'+'lam_'+str(args.lambda_) +'_'+str(args.experiment_no)+'_'+'local_epoch_'+str(args.num_local_epochs)+'_'+args.setting+'.pt')


    torch.save(trust_weight_dict,trust_weight_path)
    torch.save(test_accuracy_dict,test_acc_path)
    torch.save(ref_accuracy_dict,ref_acc_path)
    return 1



