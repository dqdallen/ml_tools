from torch.utils.data import Dataset
import pandas as pd
import random
import numpy as np
import torch
# from sklearn.datasets import fetch_kddcup99, fetch_covtype

class KDDCup99Dataset(Dataset):
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.csv_data = pd.read_csv(data_path, header=None, nrows=300000)
        self.csv_data = self.csv_data.fillna(0)
        self.data = self.csv_data.iloc[:,:-1]
        self.target = self.csv_data.iloc[:,-1]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = torch.tensor(self.data.iloc[idx, :].tolist())
        label = torch.tensor(self.target.iloc[idx].tolist())
        return sample, label


#定义kdd99数据预处理函数
def preprocess_kddcup99():
    source_file='./transformers/datasets/data/kddcupdata.csv'
    handled_file='./transformers/datasets/data/kddcup.csv'
    source_data = pd.read_csv(source_file, header=None)

    source_data[1] = source_data[1].replace({'tcp': 0, 'udp': 1, 'icmp': 2})
    service  = ['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
    service_ind = list(range(len(service)))
    service_dict = dict(zip(service, service_ind))
    source_data[2] = source_data[2].replace(service_dict)
    flag = ['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    flag_ind = list(range(len(flag)))
    flag_dict = dict(zip(flag, flag_ind))
    source_data[3] = source_data[3].replace(flag_dict)
    label = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    'spy.', 'rootkit.']

    label_ind = list(range(len(label)))
    label_dict = dict(zip(label, label_ind))
    source_data[41] = source_data[41].replace(label_dict)

    source_data.to_csv(handled_file, index=False, header=False)
    cate_cols = [1, 2, 3, 6, 11, 13, 14, 20, 21]
    return cate_cols

def num_minmax(path, cate_cols):
    source_file = './transformers/datasets/data/kddcup.csv'
    train_file = './transformers/datasets/data/kddcup_train.csv'
    valid_file = './transformers/datasets/data/kddcup_valid.csv'
    test_file = './transformers/datasets/data/kddcup_test.csv'
    
    data = pd.read_csv(path, header=None)
    ind = list(range(data.shape[0]))
    random.shuffle(ind)
    train_ind = ind[:int(0.8*len(ind))]
    valid_ind = ind[int(0.8*len(ind)):int(0.9*len(ind))]
    test_ind = ind[int(0.9*len(ind)):]
    train_data = data.iloc[train_ind, :]
    train_label = train_data.iloc[:, -1]
    valid_data = data.iloc[valid_ind, :]
    valid_label = valid_data.iloc[:, -1]
    test_data = data.iloc[test_ind, :]
    test_label = test_data.iloc[:, -1]

    num_cols = list(set(range(41)) - set(cate_cols))
    num_train_data = train_data.iloc[:, num_cols]
    cate_train_data = train_data.iloc[:, cate_cols]
    num_valid_data = valid_data.iloc[:, num_cols]
    cate_valid_data = valid_data.iloc[:, cate_cols]
    num_test_data = test_data.iloc[:, num_cols]
    cate_test_data = test_data.iloc[:, cate_cols]
    minn = num_train_data.min()
    maxx = num_train_data.max()
    num_train_data = (num_train_data - minn) / (maxx - minn)
    train_data = pd.concat([cate_train_data, num_train_data, train_label], axis=1)
    num_valid_data = (num_valid_data - minn) / (maxx - minn)
    valid_data = pd.concat([cate_valid_data, num_valid_data, valid_label], axis=1)
    num_test_data = (num_test_data - minn) / (maxx - minn)
    test_data = pd.concat([cate_test_data, num_test_data, test_label], axis=1)
    train_data.to_csv(train_file, index=False, header=False)
    valid_data.to_csv(valid_file, index=False, header=False)
    test_data.to_csv(test_file, index=False, header=False)

#num_minmax('./transformers/datasets/data/kddcup.csv', [1, 2, 3, 6, 11, 13, 14, 20, 21])
# train_file = './transformers/datasets/data/kddcupdata.csv'
# print(pd.read_csv(train_file, header=None).max())