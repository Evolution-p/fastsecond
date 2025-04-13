# -*- coding:utf8 -*-
import os
class BatchRename():
  '''
  批量重命名文件夹中的图片文件
  '''
  def __init__(self):
    self.path = '/home/admin715/Desktop/Unet-Segmentation-Pytorch-Nest-of-Unets-master/data/new_img5/'
  def rename(self):
    filelist = os.listdir(self.path)
    total_num = len(filelist)
    j = 0
    for item in filelist:
      if item.endswith('.png'):
        src = os.path.join(os.path.abspath(self.path), item)
        #dst = os.path.join(self.path, "%d.png" % j)
        #dst = os.path.join(self.path, "%03d.png" % j)
        dst = os.path.join(self.path, "%03d.png" % j)
        #dst = os.path.join(self.path,"%03d_mask.png"%j)
        j = j + 1

        os.rename(src, dst)




if __name__ == '__main__':
  demo = BatchRename()
  demo.rename()