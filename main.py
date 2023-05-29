from client import Client
import torchvision
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader
import json
from data_utils import PDLDataSet, get_ref_data, dl_to_sampler, SharedData, save_results
import os
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('******* the device used is:', device)
print('num of gpus:',torch.cuda.device_count())
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dspath", "--dataset_path", default ='/mlodata1/dongyang/datasets/', type=str)
parser.add_argument("-expno", "--experiment_no", default =0, type=int)
parser.add_argument("-seed", "--seed", default =11, type=int)
parser.add_argument("-nc", "--num_clients", default =10, type=int)
parser.add_argument("-gr", "--num_global_rounds", default = 50, type=int)
parser.add_argument('-le',"--num_local_epochs",default=5, type=int)
parser.add_argument('-lr',"--learning_rate",default=5e-3,type=float)
parser.add_argument('-mom',"--momentum",default=0.9)
parser.add_argument('-lam','--lambda_',default=0.5,type=float)
parser.add_argument('-ncl','--num_classes',default=10, type=int)
parser.add_argument('-nch','--num_channels',default=3, type=int)
parser.add_argument('-trust','--trust_update',default='dynamic', type=str,help='static, dynamic, naive')
parser.add_argument('-consensus','--consensus_mode',default='soft_assignment', type=str,help='majority_voting, soft_assignment')
parser.add_argument('-ds','--dataset_name',type=str, 
                    default='Cifar10',help='MNIST, Cifar10, Cifar100, fed-isic-2019')
parser.add_argument('-gpuids','--device_ids',type=list, 
                    default=[0])
parser.add_argument('-device','--device',type=str, 
                    default='cuda')
parser.add_argument('-train_bs','--train_batch_size',type=int, default=64)
parser.add_argument('-ref_bs','--ref_batch_size',type=int, default=256)
parser.add_argument('-test_bs','--test_batch_size',type=int,default=256)                  
parser.add_argument('-sim','--sim_measure',type=str,default='cosine',help='[true_label,cosine]')
parser.add_argument('-prer','--pretraining_rounds',type=int,default=5)
parser.add_argument('-cmode','--cmode',type=str,default='regularized')
parser.add_argument('-setting','--setting',type=str,default ='normal',help='choose between [2sets, evil,normal]')
# parser.add_argument('-sampler','--sample_ratio',type=float,default =1.0,help='sample shared data to fasten training process')
parser.add_argument('-respath','--res_path',type=str,default='/mlodata1/dongyang/results/res_april/lam-search/')
parser.add_argument('-arch_name','--arch_name',type=str,default='efficientnet-b0',help='only when selects fedisic')
parser.add_argument('-metric','--metric',type=str,default='acc',help='choose between bacc and acc')
parser.add_argument('-trust_freq','--trust_update_frequency',type=int,default=1, help='how often should trust be updated')
args = parser.parse_args()
 
print('dataset used:', args.dataset_name)
print('setting:',args.setting)

workers = {}
torch.manual_seed(args.seed)
np.random.seed(args.seed)

ref_datapath = os.path.join(args.dataset_path,args.dataset_name,'ref')#
ref_X, ref_y = get_ref_data(ref_datapath)
ref_loader = DataLoader(SharedData(ref_X, ref_y),batch_size=args.ref_batch_size,
             shuffle=False,pin_memory=True, num_workers=0)
# torch.save(ref_y,'/mlodata1/dongyang/results/res/ref_Y_'+args.dataset_name+'.pt')

respath = os.path.join(args.res_path,args.dataset_name)
if not os.path.exists(respath):
    os.mkdir(respath)

clients_sample_size = np.empty(args.num_clients)

if args.setting == 'normal':
    if args.dataset_name[0:7] =='Cifar10':
        for i in range(0,args.num_clients,1):
            workers[i] = Client(i, model_idx='resnet', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
    elif args.dataset_name in ['fed-isic-2019','fed-isic-2019-new']:
        for i in range(0,args.num_clients,1):
            workers[i] = Client(i, model_idx='effecient-net', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
    elif args.dataset_name in ['MNIST']:
        for i in range(0,args.num_clients,1):
            workers[i] = Client(i, model_idx='cnn', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()


elif args.setting == '2sets':
    if args.dataset_name[0:7] =='Cifar10':
        for i in range(0,5,1):
            workers[i] = Client(i, model_idx='resnet', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
        for i in range(5,10,1):
            workers[i] = Client(i, model_idx='fnn', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
    elif args.dataset_name in ['fed-isic-2019','fed-isic-2019-new']:
        for i in range(0,3,1):
            workers[i] = Client(i, model_idx='effecient-net', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
        for i in range(3,6,1):
            workers[i] = Client(i, model_idx='fnn', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()

elif args.setting == "evil":
    client_ids = np.arange(args.num_clients)
    if args.dataset_name in ['fed-isic-2019','fed-isic-2019-new']:
        evil_idx = np.array([1,2])
        print('evil worker:',evil_idx)
        normal_idx = [id for id in client_ids if id not in evil_idx]
        for i in evil_idx:
            workers[i] = Client(i,model_idx='effecient-net', mode ='flipped',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
        for i in normal_idx:
            workers[i] = Client(i, model_idx='effecient-net', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
    elif args.dataset_name in ['Cifar100']:
        evil_idx = np.array([2,9])
        print('evil worker:',evil_idx)
        normal_idx = [id for id in client_ids if id not in evil_idx]
        for i in evil_idx:
            workers[i] = Client(i,model_idx='resnet', mode ='flipped',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
        for i in normal_idx:
            workers[i] = Client(i, model_idx='resnet', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
    elif args.dataset_name in ['Cifar10','Cifar10-uc2']:
        evil_idx = np.array([2,9])
        print('evil worker:',evil_idx)
        normal_idx = [id for id in client_ids if id not in evil_idx]
        for i in evil_idx:
            workers[i] = Client(i,model_idx='resnet', mode ='flipped',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
        for i in normal_idx:
            workers[i] = Client(i, model_idx='resnet', mode='normal',args=args)
            clients_sample_size[i] = workers[i].get_trainset_size()
    

# model training
soft_decision_dict = {}
test_accuracy_dict = {}
ref_accuracy_dict = {}
train_acc_dict = {}
trust_weight_dict = {}
ref_bacc_dict = {}
test_bacc_dict = {}

for round in range(0, args.num_global_rounds):
    test_accuracy_dict[round] = []
    soft_decision_dict[round] = []
    ref_accuracy_dict[round] = []
    train_acc_dict[round] = []
    trust_weight_dict[round] = []
    trust_weight_tmp = []
    for i in range(args.num_clients):
        if round < args.pretraining_rounds:
            soft_decision_target = None
        else:
            soft_decision_target, trust_weights = workers[i].calculate_soft_decision_target(soft_decision_dict[round-1],train_acc_dict[round-1],clients_sample_size,global_round=round)
            trust_weight_tmp.append(trust_weights)

        for epoch in range(args.num_local_epochs):
            workers[i].train(epoch, ref_loader, soft_decision_target)

        train_acc = workers[i].test(mode='train')
        train_acc_dict[round].append(train_acc)
        soft_decision_tmp = workers[i].infer_on_refdata(ref_loader)
        soft_decision_dict[round].append(soft_decision_tmp)

        ref_pred = torch.argmax(soft_decision_tmp,dim=1)
        if args.metric == 'acc':
            ref_accuracy_dict[round].append(sum(ref_pred==ref_y)/len(ref_y))
        elif args.metric == 'bacc':
            ref_accuracy_dict[round].append(balanced_accuracy_score(ref_y,ref_pred))

        test_accuracy_dict[round].append(workers[i].test(args.metric))
        tmp_path = 'models/model_weights/worker_'+str(i)+'.pt'
    
    if len(trust_weight_tmp) >0:
        trust_weight_dict[round] = torch.stack(trust_weight_tmp)

    print('round %i finished'% round)
    print('local accuracy after round',str(round),':',np.mean(test_accuracy_dict[round]))
    print('local accuracy:',test_accuracy_dict[round])
    print('global accuracy after round',str(round),':',np.mean(ref_accuracy_dict[round]))
    print('global acc:',ref_accuracy_dict[round])

if save_results(trust_weight_dict,test_accuracy_dict,ref_accuracy_dict, respath, args) == 1:
    print('saved file successfully! The results are under', args.res_path)


