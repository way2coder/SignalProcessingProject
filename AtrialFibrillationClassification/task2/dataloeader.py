import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import scipy.io as io
import torch
import matplotlib.pyplot as plt 

# Random clipping has been implemented, 
# and you need to add noise and random scaling. 
# Generally, the scaling should be done before the crop.
# In general, do not add scaling and noise enhancement options during testing

class ECG_dataset(Dataset):
    def __init__(self, base_file):
        self.file_list=[]
        self.base_file=base_file
        
        for i in range(5):
            data=pd.read_csv(base_file+'cv/cv'+str(i)+'.csv')
            self.file_list.append(data.to_numpy())
        self.file=None
        self.file=self.file_list[0]
        for i in range(0, 4):
            self.file=np.append(self.file, self.file_list[i], axis=0)

    def __len__(self):
        return self.file.shape[0]

    def load_data(self,file_name,label):
        #读取数据
        mat_file = self.base_file+'/training2017/'+file_name+'.mat'
        data = io.loadmat(mat_file)['val']
        if label=='N':
            one_hot=torch.tensor([0])
        elif label=='O':
            one_hot=torch.tensor([0])
        elif label=='A':
            one_hot=torch.tensor([1])
        elif label=='~':
            one_hot=torch.tensor([0])
            
        return data, one_hot

    def crop_padding(self, data, time):
        #随机crop
        if data.shape[0] <= time:
            data=np.pad(data, (0,time-data.shape[0]), 'constant')
        elif data.shape[0] > time:
            end_index = data.shape[0]-time
            start = np.random.randint(0, end_index)
            data = data[start:start+time]

        return data

    def data_process(self, data):
        # 学习论文以及数据集选择合适和采样率
        # 并完成随机gaussian 噪声和随机时间尺度放缩
        data=data[::3]
        data=data-data.mean()
        data=data/data.std()
        data=self.crop_padding(data,2400)
        data=torch.tensor(data)

        return data

    def __getitem__(self, idx):
        file_name = self.file[idx][1]
        label = self.file[idx][2]
        data, one_hot = self.load_data(file_name, label)
        data = self.data_process(data[0]).unsqueeze(0).float()
        one_hot = one_hot.unsqueeze(0).float()

        return data, one_hot, file_name
