from PIL import Image 
from torch.utils.data import Dataset 
import cv2
import os 
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms 
import torchvision 
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import scipy

BLACK = [0,0,0]
png_names = [] 
file_name = './ReHo_image_png'
dir_list = os.listdir(file_name)
print(dir_list)
for i in dir_list:
    dir_name = os.path.join(file_name,i)
    for j in os.listdir(dir_name):
        png_names.append(os.path.join(dir_name,j))

data_x = png_names
data_y = [] 
for i in data_x:
    if i.split('\\')[1] == 'HC':
        data_y.append(0)
    elif i.split('\\')[1] == 'MCA':
        data_y.append(1)
    else:
        data_y.append(2)

X_train, X_test, y_train,y_test = train_test_split(data_x,data_y,test_size=0.25,random_state=0)

class MriDataset(Dataset):
    def __init__(self,mode,transform):
        self.transform = transform 
        self.fnames = [] 
        self.labels = [] 
        if mode == 'train':
            self.fnames = X_train
            self.labels = y_train 
            
        elif mode == 'val':
            self.fnames = X_test
            self.labels = y_test
        self.num_samples = len(self.labels)

    def __getitem__(self,idx):
        '''
        load image
        '''
        fname = self.fnames[idx] 
        img = Image.open(fname)
        img = img.resize((32,32))
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[idx] 
        return img,label 

    def __len__(self):
        return self.num_samples

# Data
# print('==> Preparing data..')
# transform_train = transforms.Compose([
#     #transforms.Resize(64), #把给定的图片resize到given size
#     #transforms.RandomCrop(64, padding=4), #在图片的中间区域进行裁剪，size可以是tuple，(target_height, target_width)。size也可以是一个Integer，在这种情况下，切出来的图片的形状是正方形。
#     #transforms.RandomHorizontalFlip(), #以0.5的概率水平翻转给定的PIL图像
#     transforms.ToTensor(),# range [0, 255] -> [0.0,1.0]
#     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])
# trainset = MriDataset(mode='train',transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=1)

def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    plt.imshow(inp)
    plt.pause(1)
      # pause a bit so that plots are updated

def imshow2(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    scipy.misc.imsave('test.png', inp)


# if __name__ == '__main__':
#     inputs, classes = next(iter(trainloader))

#     # Make a grid from batch
#     out = torchvision.utils.make_grid(inputs)
#     imshow(out)
