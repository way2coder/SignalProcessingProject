
import torch
import cv2
import os
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms


#定义网络
""" Parts of the U-Net model """
class Conv_Block(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv_Block, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,1,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

# 下采样
class DownSample(nn.Module):
    def __init__(self,channel):
        super(DownSample, self).__init__()
        self.layer=nn.Sequential(
            nn.Conv2d(channel,channel,3,2,1,padding_mode='reflect',bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.layer(x)

# 上采样
class UpSample(nn.Module):
    def __init__(self,channel):
        super(UpSample, self).__init__()
        self.layer=nn.Conv2d(channel,channel//2,1,1)
    def forward(self,x,feature_map):
        up=F.interpolate(x,scale_factor=2,mode='nearest')
        out=self.layer(up)
        return torch.cat((out,feature_map),dim=1)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.c1=Conv_Block(1,64)
        self.d1=DownSample(64)
        self.c2=Conv_Block(64,128)
        self.d2=DownSample(128)
        self.c3=Conv_Block(128,256)
        self.d3=DownSample(256)
        self.c4=Conv_Block(256,512)
        self.d4=DownSample(512)
        self.c5=Conv_Block(512,1024)
        self.u1=UpSample(1024)
        self.c6=Conv_Block(1024,512)
        self.u2 = UpSample(512)
        self.c7 = Conv_Block(512, 256)
        self.u3 = UpSample(256)
        self.c8 = Conv_Block(256, 128)
        self.u4 = UpSample(128)
        self.c9 = Conv_Block(128, 64)
        self.out=nn.Conv2d(64,1,3,1,1)
        self.Th=nn.Sigmoid()

    def forward(self,x):
        R1=self.c1(x)
        R2=self.c2(self.d1(R1))
        R3 = self.c3(self.d2(R2))
        R4 = self.c4(self.d3(R3))
        R5 = self.c5(self.d4(R4))
        O1=self.c6(self.u1(R5,R4))
        O2 = self.c7(self.u2(O1, R3))
        O3 = self.c8(self.u3(O2, R2))
        O4 = self.c9(self.u4(O3, R1))

        return self.Th(self.out(O4))
    


transform=transforms.Compose([
    transforms.ToTensor()
])
class MyDataset(Dataset):

    def __init__(self,data_file = None, mask_files = None,data_folder_path = None, mask_folder_path = None, test_flag = False):
        super(MyDataset).__init__()
        self.data_file = data_file
        self.mask_files = mask_files
        self.data_folder_path = data_folder_path
        self.mask_folder_path = mask_folder_path
        self.test_flag = test_flag

    
    def __len__(self):
        return len(self.data_file)

    # cv读取图像信息
    def __getitem__(self,idx):   
        fname = self.data_file[idx]
        im = cv2.imread(os.path.join(self.data_folder_path,fname),cv2.IMREAD_GRAYSCALE)
        # im = cv2.equalizeHist(im)   #直方图均衡化，提高对比度。
        mname = self.mask_files[idx]
        segemnt_im = cv2.imread(os.path.join(self.mask_folder_path,mname),cv2.IMREAD_GRAYSCALE)
        # segemnt_im = cv2.equalizeHist(segemnt_im)
        #对灰度值进行归一化
        if self.test_flag:
            return transform(im/255).float(),transform(segemnt_im/255).float(),fname
        return transform(im/255).float(),transform(segemnt_im/255).float()
