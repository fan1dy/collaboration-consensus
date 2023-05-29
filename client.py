import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from efficientnet_pytorch import EfficientNet
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data_utils import PDLDataSet,get_ref_labels_by_client,SharedData,dl_to_sampler
from models.netcnn import NetCNN
from models.netfnn import NetFNN
from models.resnet import resnet20
import os
import time
import numpy.random as random
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

def cross_entropy(input, target):
    return torch.mean(-torch.sum(target * torch.log(input+1e-8), 1))

class cosine_similarity(nn.Module):
    def __init__(self,mode='regularized'):
        super(cosine_similarity, self).__init__()
        self.mode = mode
    def forward(self,input, target):
        sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        simi = sim(input,target)
        if self.mode == 'regularized':
            ent = -torch.sum(target*torch.log(target+1e-8),dim=1)+1.0
            # prevent entropy being too small
            reg_sim = torch.div(simi,ent)
        elif self.mode == 'normal':
            reg_sim = simi
        return torch.mean(reg_sim)

class true_label_similarity(nn.Module):
    def __init__(self,id, labels, num_clients,device):
        super(true_label_similarity, self).__init__()
        self.id = id
        self.labels = labels
        self.lengs = [len(self.labels[id]) for id in range(num_clients)]
        self.device = device
    def forward(self,input, target):
        true_lbs = torch.Tensor(self.labels[self.id]).to(self.device)
        idx_start = int(0+ sum(self.lengs[:self.id]))
        idx_end = int(sum(self.lengs[:self.id+1]))
        pred_lbs = torch.argmax(input,dim=1)[idx_start:idx_end]
        return torch.sum(true_lbs==pred_lbs)/len(true_lbs)

class Client(object):
    def __init__(self, worker_index, model_idx='resnet', mode='normal',args={}):
        self.dataset = args.dataset_name
        if self.dataset[0:7] == 'Cifar10':
            if model_idx == 'resnet':
                self.model = resnet20(num_classes=args.num_classes)
            elif model_idx == 'cnn': 
                self.model = NetCNN(in_features=args.num_channels, num_classes=args.num_classes,dim=1600)
            elif model_idx == 'fnn':
                self.model = NetFNN(input_dim=3*32*32, mid_dim=100, num_classes=args.num_classes)
        elif self.dataset in ['MNIST']:
            self.model = NetCNN(in_features=1, num_classes=10,dim=1024)
        elif self.dataset in ['fed-isic-2019','fed-isic-2019-new']:
            if model_idx == 'effecient-net':
                self.model =  EfficientNet.from_pretrained(args.arch_name, num_classes=args.num_classes)
            elif model_idx == 'cnn':
                self.model = NetCNN(in_features=args.num_channels, num_classes=args.num_classes,dim=10816)
            elif model_idx == 'fnn':
                self.model = NetFNN(input_dim=3*64*64, mid_dim=100, num_classes=8)

        self.device = args.device
        self.consensus = args.consensus_mode
        self.id = worker_index
        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.num_classes = args.num_classes
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.trust = args.trust_update
        self.datapath = os.path.join(args.dataset_path,self.dataset)
        self.ref_labels = get_ref_labels_by_client(os.path.join(self.datapath,'ref'))
        if args.sim_measure =='cosine':
            self.sim = cosine_similarity(args.cmode) 
            self.task_difficulty_switch = False
        elif args.sim_measure == 'true_label':
            self.sim = true_label_similarity(self.id,self.ref_labels,args.num_clients,self.device)
            self.task_difficulty_switch = False
        with open(os.path.join(self.datapath,'train',str(self.id)+'.npz'), 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()
        with open(os.path.join(self.datapath,'test',str(self.id)+'.npz'), 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()
        self.train_dataset = PDLDataSet(train_data,mode = mode, num_class=self.num_classes)
        self.test_dataset = PDLDataSet(test_data, mode ='normal')
        #train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=True) 
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size,shuffle=True,pin_memory=True, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)
        self.lambda_ = args.lambda_
        self.trust_weights = []
        self.num_clients = args.num_clients
        self.period = args.trust_update_frequency
        self.prer = args.pretraining_rounds

    def train(self,epoch, extra_data,soft_decision_target=None):
        ref_sample = dl_to_sampler(extra_data)
        self.model.to(self.device)
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            with torch.no_grad():
                extra_data_sampled, _, ref_idx = ref_sample()
                extra_data_sampled = extra_data_sampled.to(self.device)
            soft_decision = F.softmax(self.model(extra_data_sampled),dim=1)
            local_loss = torch.nn.CrossEntropyLoss()(output, target)
            if soft_decision_target == None: 
                soft_decision_loss = 0 
            else:
                soft_decision_target = soft_decision_target.to(self.device)
                soft_decision_loss = cross_entropy(soft_decision,soft_decision_target[ref_idx,:])
            loss = local_loss + self.lambda_ *soft_decision_loss
            loss.backward()
            self.optimizer.step()
            

    @torch.no_grad()
    def infer_on_refdata(self,ref_loader):
        soft_decision_list = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (data,_,_) in enumerate(ref_loader):
                data = data.to(self.device)
                soft_decision = F.softmax(self.model(data),dim=1)
                soft_decision_list.append(soft_decision)
        soft_decisions = torch.cat(soft_decision_list,dim=0)
        return soft_decisions.detach().cpu()
    
    @torch.no_grad()
    def soft_assignment_to_hard_assignment(self,soft_decisions):
        a  = torch.argmax(soft_decisions,1)
        hard_decisions = torch.zeros_like(soft_decisions)
        for i in range(hard_decisions.shape[0]):
            hard_decisions[i,a[i]]=1
        return hard_decisions
    
    @torch.no_grad()
    def calculate_trust_weight_mat(self, soft_decision_list,train_acc_list,clients_sample_size):
        sd_eigen = soft_decision_list[self.id]
        sim_list = []
        for j in range(len(soft_decision_list)):
            sd_other = soft_decision_list[j]
            sim_ = self.sim(sd_other.to(self.device),sd_eigen.to(self.device))
            sim_list.append(sim_)
        sim_list = torch.tensor(sim_list)
        if train_acc_list ==[]:
            pass 
        else:
            if self.task_difficulty_switch == True:
                local_loss_list = (1-torch.Tensor(train_acc_list)+1e-5)/clients_sample_size
                local_loss_list -= torch.min(local_loss_list)
                local_loss_list /= torch.max(local_loss_list)
                local_loss_list += 1
                sim_list = sim_list/local_loss_list
 
        trust_weights = sim_list/torch.sum(sim_list)
        self.trust_weights = trust_weights
        return trust_weights


    @torch.no_grad()
    def calculate_soft_decision_target(self,soft_decision_list,train_acc_list,clients_sample_size,global_round):
        if self.consensus == 'soft_assignment':
            if self.trust == 'static':
                if self.trust_weights == []:
                    self.trust_weights = self.calculate_trust_weight_mat( soft_decision_list,train_acc_list,clients_sample_size)
                elif self.trust_weights != []:
                    pass
               
            elif self.trust == 'dynamic':
                if (global_round-self.prer ) % self.period == 0 :
                    self.trust_weights = self.calculate_trust_weight_mat( soft_decision_list,train_acc_list,clients_sample_size)
                else:
                    pass
               
            elif self.trust == 'naive':
                self.trust_weights = torch.ones(self.num_clients,1)
                self.trust_weights =  self.trust_weights/self.num_clients

            weighted_soft_decisions = [self.trust_weights[i]*soft_decision_list[i] for i in range(self.num_clients)]
            target = sum(weighted_soft_decisions)
            target = torch.nn.functional.normalize(target,p=1.0,dim=1)
            return target, self.trust_weights
        
        elif self.consensus == 'majority_voting':
            hard_labels = [self.soft_assignment_to_hard_assignment(soft_decision_list[i]) for i in range(self.num_clients)]
            target = sum(hard_labels)
            target = self.soft_assignment_to_hard_assignment(target)
            self.trust_weights = torch.ones(self.num_clients,1)/self.num_clients
            return target, self.trust_weights

    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
    
    @torch.no_grad()
    def test(self,metric='acc',mode='test'):
        if mode == 'train':
            loader = self.train_loader
        elif mode == 'test':
            loader=self.test_loader

        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            if metric == 'acc':
                preds = []
                for batch_idx, (data, target) in enumerate(loader):
                    data = data.to(self.device)
                    target = target.to(self.device)
                    pred_test = torch.sum(torch.argmax(self.model(data),dim=1)==target)
                    preds.append(pred_test.detach().cpu())
                return np.sum(preds)/len(loader.dataset)
            elif metric == 'bacc':
                pred_test_list = []
                target_list = []
                for batch_idx, (data, target) in enumerate(loader):
                    data = data.to(self.device)
                    target_list.extend(target) 
                    pred_test = torch.argmax(self.model(data),dim=1).detach().cpu()
                    pred_test_list.append(pred_test)
                pred_test = torch.cat(pred_test_list,0)
                bacc = balanced_accuracy_score(target_list,pred_test)
                return bacc

    def get_trainset_size(self):
        return len(self.train_loader.dataset)