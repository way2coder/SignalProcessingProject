import cv2
import numpy as np
import os
import pandas as ps
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
# This is for the progress bar.
from tqdm.auto import tqdm
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image
from model import UNet, MyDataset
from torch.utils.data import random_split
from my_metrics import calculate_metrics
import csv

if __name__ == '__main__':
    #设置设备，选择cuda
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print('mps' if torch.backends.mps.is_available() else 'cpu')

    weight_path='params/unet.pth'
    model = UNet().to(device)
    
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path))
        print('successful load weight！\n')
    else:
        print('model is not exist!\n')
        exit()

    # data_folder_path = '../data/FAZ/Domain1/test/imgs'
    # mask_folder_path = '../data/FAZ/Domain1/mask/mask'

    # data_files = os.listdir(data_folder_path)
    # mask_files= os.listdir(mask_folder_path)


    # dataset = MyDataset(data_file=data_files,mask_files=mask_files,data_folder_path=data_folder_path,mask_folder_path=mask_folder_path)


    for i in range(1,6):
        data_folder_path = f'../data/FAZ/Domain{i}/test/imgs'
        mask_folder_path = f'../data/FAZ/Domain{i}/test/mask'
        save_path = f"result/Domain{i}"
        if not os.path.isdir('result'):
            os.mkdir('result')
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        data_files = os.listdir(data_folder_path)
        mask_files= os.listdir(mask_folder_path)
        dataset = MyDataset(data_file=data_files,mask_files=mask_files,data_folder_path=data_folder_path,mask_folder_path=mask_folder_path)

        datalen = len(dataset)
        testLoader = DataLoader(dataset,batch_size=8,shuffle=True)
        [all_dc,all_jc,all_hd,all_assd,all_sp,all_recall,all_pre] = [0,0,0,0,0,0,0]
        model.eval()
        print(f"start testing domain{i}\n")
        cur = 1
        
        for idx,(img,seg_img) in enumerate(testLoader):
            img = img.to(device)
            seg_img = seg_img.to(device)
            output = model(img)
            
            for pred,target in zip(output, seg_img):
                dice,jaccard,hd95_score,assd_score,sp_score,recall_score,pre_score = calculate_metrics(pred,target)
                all_dc += dice/datalen
                all_jc += jaccard/datalen
                all_hd += hd95_score/datalen
                all_assd += assd_score/datalen
                all_sp += sp_score/datalen
                all_recall += recall_score/datalen
                all_pre += pre_score/datalen
                save_image(pred,f'{save_path}/{cur}.png')
                cur = cur + 1
        print(f"the metrics of model on Domain{i} are as below\n")
        print(all_dc,all_jc,all_hd,all_assd,all_sp,all_recall,all_pre)
    
    print("finish testing\n")

    
    # testDataset = ConcatDataset([setlist[0],setlist[1],setlist[2],setlist[3],setlist[4]])
    # datalen = len(testDataset)
    # testLoader = DataLoader(testDataset,batch_size=datalen,shuffle=True)
    # model.eval()
    # [all_dc,all_jc,all_hd,all_assd,all_sp,all_recall,all_pre] = [0,0,0,0,0,0,0]
    # for idx,(img,seg_img) in enumerate(testLoader):
        # img = img.to(device)
        # seg_img = seg_img.to(device)
        # output = model(img)
        # for pred,target in zip(valid_output,valid_seg_img):
        #     dice,jaccard,hd95_score,assd_score,sp_score,recall_score,pre_score = calculate_metrics(pred,target)
        #     all_dc += dice/datalen
        #     all_jc += jaccard/datalen
        #     all_hd += hd95_score/datalen
        #     all_assd += assd_score/datalen
        #     all_sp += sp_score/datalen
        #     all_recall += recall_score/datalen
        #     all_pre += pre_score/datalen
    
    

