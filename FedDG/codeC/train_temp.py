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
    data_folder_path = '../data/FAZ/Domain1/train/imgs'
    mask_folder_path = '../data/FAZ/Domain1/train/mask'

    data_files = os.listdir(data_folder_path)
    mask_files= os.listdir(mask_folder_path)


    dataset = MyDataset(data_file=data_files,mask_files=mask_files,data_folder_path=data_folder_path,mask_folder_path=mask_folder_path)
    weight_path='params/unet.pth'
    #data_path=r'data'
    model = UNet().to(device)
    # if os.path.exists(weight_path):
    #     net.load_state_dict(torch.load(weight_path))
    #     print('successful load weight！')
    # else:
    #     print('not successful load weight')
    train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[195, 49], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True)
    valid_loader = DataLoader(valid_dataset,batch_size=8,shuffle=True)
    opt = optim.Adam(model.parameters())
    loss_fun = nn.BCELoss()
    all_epoch = 50
    
    lossArr = []
    dicelist = []
    jclist = []
    hd95list = []
    assdlist = []
    splist = []
    recalllist = []
    prelist = []

    prev_dice = 0
    print("model starts training\n")
    for current_epoch in range(all_epoch):
        model.train()
        lossItem = 0
        for idx, (image,segment_image) in enumerate(train_loader):
            image = image.to(device)
            segment_image = segment_image.to(device)
            opt.zero_grad()
            output_img = model(image)
            train_loss = loss_fun(output_img, segment_image)
            train_loss.backward()
            opt.step()
            lossItem = lossItem + train_loss.item()
        print(f"finished the epoch {current_epoch} training!\n")

        model.eval()
        [all_dc,all_jc,all_hd,all_assd,all_sp,all_recall,all_pre] = [0,0,0,0,0,0,0]
        datalen = len(valid_dataset)
        for idx,(valid_img,valid_seg_img) in enumerate(valid_loader):
            valid_img = valid_img.to(device)
            valid_seg_img = valid_seg_img.to(device)
            valid_output = model(valid_img)
            for pred,target in zip(valid_output,valid_seg_img):
                dice,jaccard,hd95_score,assd_score,sp_score,recall_score,pre_score = calculate_metrics(pred,target)
                all_dc += dice/datalen
                all_jc += jaccard/datalen
                all_hd += hd95_score/datalen
                all_assd += assd_score/datalen
                all_sp += sp_score/datalen
                all_recall += recall_score/datalen
                all_pre += pre_score/datalen
        print(f'epoch{current_epoch}--train_loss = {lossItem},dice_score = {all_dc}')
        if not os.path.isdir("params"):
            os.mkdir("params")
        # if all_dc - prev_dice < -0.05:
        #     print(f"training stop at epoch {current_epoch} for too little improment on dice score!\n")
        #     break
        torch.save(model.state_dict(),weight_path)
        lossArr.append(lossItem)
        dicelist.append(all_dc)
        jclist.append(all_jc)
        hd95list.append(all_hd)
        assdlist.append(all_assd)
        splist.append(all_sp)
        recalllist.append(all_recall)
        prelist.append(all_pre)
        # prev_dice = all_dc
    final_dice = dicelist[-1]
    
    print(f"model finished training with a dice on validation dataset:{final_dice}\n")
    transposed_lists = zip(lossArr, dicelist, jclist, hd95list, assdlist, splist, recalllist, prelist)

    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in transposed_lists:
            writer.writerow(row)

