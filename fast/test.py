from __future__ import print_function, division
import os
import numpy as np
from PIL import Image
import glob
#import SimpleITK as sitk
from torch import optim
import torch.utils.data
import torch
import torch.nn.functional as F

import torch.nn
import torchvision
import matplotlib.pyplot as plt
import natsort
from torch.utils.data.sampler import SubsetRandomSampler
from Data_Loader import Images_Dataset, Images_Dataset_folder
import torchsummary
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import shutil
import random
from Models import Unet_dict, NestedUNet,NestedUNet_att, NestedUNet_mult_att,U_Net, R2U_Net, AttU_Net, R2AttU_Net
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
from ploting import plot_kernels, LayerActivations, input_images, plot_grad_flow
from Metrics import dice_coeff, accuracy_score
import time
import csv

train_on_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if train_on_gpu else "cpu")

model_Inputs = [U_Net, R2U_Net, AttU_Net, R2AttU_Net, NestedUNet,NestedUNet_att,NestedUNet_mult_att]

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test

data_transform = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),
         #   torchvision.transforms.CenterCrop(96),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



batch_size = 4
print('batch_size = ' + str(batch_size))

valid_size = 0.15

epoch = 200
print('epoch = ' + str(epoch))

random_seed = random.randint(1, 100)
print('random_seed = ' + str(random_seed))

shuffle = True
valid_loss_min = np.Inf
num_workers = 4
lossT = []
lossL = []
lossL.append(np.inf)
lossT.append(np.inf)
epoch_valid = epoch-2
n_iter = 1
i_valid = 0

pin_memory = False
if train_on_gpu:
    pin_memory = True

model_test = model_unet(model_Inputs[4], 3, 1)

model_test.to(device).half()

model_test.load_state_dict(torch.load('./model3_NestedUNet/NestedUNet_' +
                   str(epoch) + '_' + str(batch_size)+ '/Unet_epoch_' + str(epoch)
                   + '_batchsize_' + str(batch_size) + '.pth'))

model_test.eval()

#######################################################
#opening the test folder and creating a folder for generated images
#######################################################
test_folderP = './data/test2/*'
test_folderL = './data/test_GT2/*'

read_test_folder = glob.glob(test_folderP)
x_sort_test = natsort.natsorted(read_test_folder)  # To sort


read_test_folder112 = './model3_NestedUNet/gen_images1'


if os.path.exists(read_test_folder112) and os.path.isdir(read_test_folder112):
    shutil.rmtree(read_test_folder112)

try:
    os.mkdir(read_test_folder112)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder112)
else:
    print("Successfully created the testing directory %s " % read_test_folder112)


#For Prediction Threshold

read_test_folder_P_Thres = './model3_NestedUNet/pred_threshold1'


if os.path.exists(read_test_folder_P_Thres) and os.path.isdir(read_test_folder_P_Thres):
    shutil.rmtree(read_test_folder_P_Thres)

try:
    os.mkdir(read_test_folder_P_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_P_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_P_Thres)

#For Label Threshold

read_test_folder_L_Thres = './model3_NestedUNet/label_threshold1'


if os.path.exists(read_test_folder_L_Thres) and os.path.isdir(read_test_folder_L_Thres):
    shutil.rmtree(read_test_folder_L_Thres)

try:
    os.mkdir(read_test_folder_L_Thres)
except OSError:
    print("Creation of the testing directory %s failed" % read_test_folder_L_Thres)
else:
    print("Successfully created the testing directory %s " % read_test_folder_L_Thres)




#######################################################
#saving the images in the files
#######################################################

img_test_no = 0

for i in range(len(read_test_folder)):
    im = Image.open(x_sort_test[i])

    im1 = im
    im_n = np.array(im1)
    im_n_flat = im_n.reshape(-1, 1)

    for j in range(im_n_flat.shape[0]):
        if im_n_flat[j] != 0:
            im_n_flat[j] = 255

    s = data_transform(im)
    pred = model_test(s.unsqueeze(0).cuda().half())
    pred = F.sigmoid(pred).cpu()
    pred = pred.detach().numpy()

#    pred = threshold_predictions_p(pred) #Value kept 0.01 as max is 1 and noise is very small.

    if i % 24 == 0:
        img_test_no = img_test_no + 1

    x1 = plt.imsave('./model3_NestedUNet/gen_images1/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', pred[0][0])


####################################################
#Calculating the Dice Score
####################################################

data_transform = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),
        #    torchvision.transforms.CenterCrop(96),
             torchvision.transforms.Grayscale(),
#            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



read_test_folderP = glob.glob('./model3_NestedUNet/gen_images1/*')
x_sort_testP = natsort.natsorted(read_test_folderP)


read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort


dice_score123 = 0.0
x_count = 0
x_dice = 0
Acc = 0.0
for i in range(len(read_test_folderP)):

    x = Image.open(x_sort_testP[i])
    s = data_transform(x)
    s = np.array(s)
    s = threshold_predictions_v(s)

    #save the images
    x1 = plt.imsave('./model3_NestedUNet/pred_threshold1/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s)

    y = Image.open(x_sort_testL[i])
    s2 = data_transform(y)
    s3 = np.array(s2)
   # s2 =threshold_predictions_v(s2)

    #save the Images
    y1 = plt.imsave('./model3_NestedUNet/label_threshold1/im_epoch_' + str(epoch) + 'int_' + str(i)
                    + '_img_no_' + str(img_test_no) + '.png', s3)

    total = dice_coeff(s, s3)
    accuracy = accuracy_score(s, s3)
    #print(total)

    if total <= 0.3:
        x_count += 1
    if total > 0.3:
        x_dice = x_dice + total
    dice_score123 = dice_score123 + total
    Acc += accuracy

print('Dice Score : ' + str(dice_score123/len(read_test_folderP)))
print('Acc Score : ' + str(Acc/len(read_test_folderP)))
result_path='./model3_AttU_Net/'
f = open(os.path.join(result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([dice_score123/len(read_test_folderP),Acc/len(read_test_folderP)])
f.close()

#print(x_count)
#print(x_dice)
#print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))

