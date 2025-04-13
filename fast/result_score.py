import torchvision
import glob
import natsort
from PIL import Image
from losses import calc_loss, dice_loss, threshold_predictions_v,threshold_predictions_p
import matplotlib.pyplot as plt
import numpy as np
from Metrics import dice_coeff, accuracy_score
import csv
import os
####################################################
#Calculating the Dice Score
####################################################
epoch = 200
test_folderL = './data/test_GT/*'

data_transform = torchvision.transforms.Compose([
          #  torchvision.transforms.Resize((128,128)),
        #    torchvision.transforms.CenterCrop(96),
             torchvision.transforms.Grayscale(),
#            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



read_test_folderP = glob.glob('./model_AttU_Net/gen_images/*')
x_sort_testP = natsort.natsorted(read_test_folderP)


read_test_folderL = glob.glob(test_folderL)
x_sort_testL = natsort.natsorted(read_test_folderL)  # To sort

img_test_no = 0
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
    #x1 = plt.imsave('./model1_U_Net/pred_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    #+ '_img_no_' + str(img_test_no) + '.png', s)

    y = Image.open(x_sort_testL[i])
    s2 = data_transform(y)
    s3 = np.array(s2)
   # s2 =threshold_predictions_v(s2)

    #save the Images
    #y1 = plt.imsave('./model1_U_Net/label_threshold/im_epoch_' + str(epoch) + 'int_' + str(i)
                    #+ '_img_no_' + str(img_test_no) + '.png', s3)

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
result_path='./result_score'
f = open(os.path.join(result_path, 'result.csv'), 'a', encoding='utf-8', newline='')
wr = csv.writer(f)
wr.writerow([dice_score123/len(read_test_folderP),Acc/len(read_test_folderP)])
f.close()

#print(x_count)
#print(x_dice)
#print('Dice Score : ' + str(float(x_dice/(len(read_test_folderP)-x_count))))

