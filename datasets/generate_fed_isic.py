# datasplits and source from https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from utils import check, separate_data, split_data, save_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_clients', type=int, default = 6,
                    help='number of clients')
parser.add_argument('--dir', type=str, help='directory for storing the data',
                    default ="/mlodata1/dongyang/datasets/fed-isic-2019-new/" )
parser.add_argument('--niid',  action='store_true', help='sample non-iid data for each worker' )
parser.add_argument('--balance',action='store_true')
parser.add_argument('--partition',type=str,default='dir' )
parser.add_argument('--alpha',type=float ,default=0.1 ,help='needed when using dirichelet distr')
parser.add_argument('--refgen',action='store_true')

num_classes = 8
num_clients = 6
train_size = 0.75

transform = transforms.Compose(
        [transforms.Resize((64,64)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def split_data(X, y,ref = False,mode='byratio'):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train':[], 'test':[],'ref':[]}
    ref_data =[]
    for i in range(len(y)):
        unique, count = np.unique(y[i], return_counts=True)
        if min(count) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)
        if ref == True:
            if mode == 'byratio':
                X_test, X_ref, y_test, y_ref = train_test_split(
                        X_test,y_test, train_size=0.8, shuffle=True)
            elif mode == 'bynumber':
                #print(y_test)
                indices = np.arange(X_test.shape[0])
                sampled_idx = np.random.choice(X_test.shape[0],50,replace=False)
                #print(sampled_idx.astype(int))
                X_ref = X_test[sampled_idx,:,:,:]
                y_ref = np.array(y_test)[sampled_idx.astype(int)]
                unsampled_idx = np.setdiff1d(indices,sampled_idx)
                X_test = X_test[unsampled_idx,:,:,:]
                y_test = np.array(y_test)[unsampled_idx.astype(int)]

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))
        if ref == True:
            ref_data.append({'x': X_ref, 'y': y_ref})
            num_samples['ref'].append(len(y_ref))
    
    print("Total number of samples:", sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    if ref == True:
        print("The number of ref samples:", num_samples['ref'])
    print()
    del X, y

    return train_data, test_data, ref_data


def generate_fedisic(dir_path, num_clients, num_classes, niid, balance, partition,alpha,ref):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"
    ref_path = dir_path + 'ref/'

    if check(config_path, train_path, test_path, ref_path, num_clients, num_classes, niid, balance, partition):
        return
    '''
    download isic datasets from https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py
    and replace the following directories using your local paths
    '''
    image_dir = '/mlodata1/dongyang/datasets/fed-isic-2019/ISIC_2019_Training_Input_preprocessed'
    data = pd.read_csv("/mlodata1/dongyang/datasets/fed-isic-2019/ISIC_2019_Training_Metadata_FL.csv", delimiter= ',')
    labels = pd.read_csv("/mlodata1/dongyang/datasets/fed-isic-2019/ISIC_2019_Training_GroundTruth.csv", delimiter= ',')

    labels_new = {'image': labels['image'],
            'label': np.where(labels.iloc[:,1:]!=0)[1]}
    labels_new = pd.DataFrame(labels_new)
    num_clients = 6
    statistic = [[] for _ in range(num_clients)]
    big_table = pd.merge(data, labels_new, on='image',how="inner")
    # image,age_approx,anatom_site_general,sex,dataset
    
    num_rows = big_table.shape[0]
    client_names = np.unique(big_table['dataset'])
    image = {}
    image_label = {}
    for j in range(num_clients):
        image[j] = []
        image_label[j] = []
    statistic = [[] for _ in range(num_clients)]
    for i in tqdm(range(num_rows)):
        for j in range(num_clients):
            if big_table['dataset'][i] ==  client_names[j]:
                image_name = os.path.join(image_dir,big_table['image'][i]+'.jpg')
                img = transform(Image.open(image_name))
                image[j].append(img.numpy()) 
                image_label[j].append(big_table['label'][i])
    for j in range(num_clients):
        image[j] = np.stack(image[j], axis=0 )
        print(image[j].shape)
    for client in range(num_clients):
        for i in np.unique(image_label[client]):
            statistic[client].append((int(i), int(sum(image_label[client]==i))))
                
    train_data, test_data, ref_data = split_data(image, image_label, args.refgen, mode='bynumber')
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

    generate_fedisic(dir_path, num_clients, num_classes, niid, balance, partition,alpha,ref)