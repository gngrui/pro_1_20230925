import numpy as np
import os
import random
from skimage.io import imread
import numpy as np
import torch
import torch.nn as nn

def crop(image:np.ndarray,target_size:int)->np.ndarray:
    c,h,w=image.shape
    start_h=max(0,(h-target_size)//2)
    start_w=max(0,(w-target_size)//2)
    end_h=min(h,(h+target_size)//2)
    end_w=min(w,(w+target_size)//2)
    
    out=np.zeros((c,target_size,target_size),dtype=image.dtype)
    out[:,
        (target_size-(end_h-start_h))//2:(target_size+(end_h-start_h))//2,
        (target_size-(end_w-start_w))//2:(target_size+(end_w-start_w))//2]=\
    image[:,start_h:end_h,start_w:end_w]
    return out

def split_dataset():
    images_path=r'C:\Users\gengrui\workspace\pro_1_20230925\kvasir-seg\images'
    masks_path=r'C:\Users\gengrui\workspace\pro_1_20230925\kvasir-seg\masks'
    file_name=[y.split('.')[0] for y in os.listdir(images_path)]
    n=len(file_name)
    train_dataset=file_name[:int(n*0.7)]
    val_dataset=file_name[int(n*0.7):int(n*0.9)]
    test_dataset=file_name[int(n*0.9):]
    return train_dataset,val_dataset,test_dataset

def mydataloader(dataset,batch_size):
    images_path=r'C:\Users\gengrui\workspace\pro_1_20230925\kvasir-seg\images'
    masks_path=r'C:\Users\gengrui\workspace\pro_1_20230925\kvasir-seg\masks'
    for i in range(0,len(dataset),batch_size):
        filenames=dataset[i:i+batch_size]
        img_array=[imread(os.path.join(images_path,y+'.jpg'))for y in filenames]
        img_array=[np.transpose(y,(2,0,1))for y in img_array]
        img_array=[crop(y,512) for y in img_array]
        mask_array=[imread(os.path.join(masks_path,y+'.jpg'))for y in filenames]
        mask_array=[np.transpose(y,(2,0,1))for y in mask_array]
        mask_array=[crop(y,324) for y in mask_array]

        img_array=np.array(img_array)
        mask_array=np.array(mask_array)
        mean=np.mean(img_array,axis=(2,3),keepdims=True)
        std=np.std(img_array,axis=(2,3),keepdims=True)
        img_array=(img_array-mean)/(std+1e-7)

        mask_array=np.where(mask_array>0,1,mask_array)
        yield (img_array, mask_array)

def my_imshow(image:np.ndarray):
    from matplotlib.pyplot import imshow
    image=image.swapaxes(0,2)
    imshow(image)

if __name__ == '__main__':
    train_dataset,val_dataset,test_dataset=split_dataset()
    train_dataloader=mydataloader(train_dataset,128)
    val_dataloader=mydataloader(val_dataset,128)
    test_dataloader=mydataloader(test_dataset,128)
    for X,Y in train_dataloader:
        print(X.shape)
        print(Y.shape)