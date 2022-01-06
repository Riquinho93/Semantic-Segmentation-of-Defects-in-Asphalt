#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

import cv2 

import glob

from UNET import UNet


def inputLoader():
    
    data = []
    N=0
    for fn in sorted(glob.glob('test3/*.jpg', recursive=False)):
    #for fn in glob.glob('/mnt/xfs/nucleus/nucleusTrain/**/*.png', recursive=True):
             print("FN== ", fn)
       # if ('images' in fn):
             img = cv2.imread(fn)
             
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
             
             img = cv2.resize(img, (320,320) )
             #print("SHAPE ", img.shape)
             
             
             img = img.astype(float)
             
             img = img/255.0
             
             img = (img-0.94)/0.37
             
             img = img.swapaxes(0,2)
             img = img.swapaxes(1,2)
            
             data.append(img)
             
             
            # f1 = cv2.flip(img, 0)
             
            # data.append(f1)
             
           #  f2 = cv2.flip(img, 1)
             
           #  data.append(f2)  

           #  f3 = cv2.flip(img, -1)
             
           #  data.append(f3)                         , 
             
             
             
             N = N + 1
   
          
    data = np.array(data)  
    data = torch.from_numpy(data)
    
     
    
    return N, data          
             
def outputLoader():
    
    data = []
    N=0
    for fn in sorted(glob.glob('testOut3/*.jpg', recursive=False)):
             
            
           #  if N<10:
           #    print(fn)
             N = N + 1   
       ### if ('images' in fn):
             img = cv2.imread(fn)
             
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting BGR to gray
             #cv2.imshow("imageTESTE",img)
             #cv2.waitKey(0)
             #print("OUTT: ", img.shape)
             img = cv2.resize(img, (320,320) )
             #cv2.imshow("img", img)
             #cv2.waitKey(0)
             #cv2.destroyAllWindows()
             #print(img)
             img[img>=1] = 255
             img[img<1] = 0
             #print(img)
             #img2 = img.copy()
             #img = img.swapaxes(0, 1)
             #cv2.imshow('IMG2= ', img2)
             #cv2.waitKey(0)
             
             data.append(img)
             
             #f1 = cv2.flip(img2, 0)
             #f1 = f1.swapaxes(0, 1)
             
            # data.append(f1)
             
             #f2 = cv2.flip(img2, 1)
             #f2 = f2.swapaxes(0, 1)
             
             #data.append(f2)
             
             #f3 = cv2.flip(img2, -1)
             #f3 = f3.swapaxes(0, 1)
             
             #data.append(f3)
             
             
             
             
    
    data = np.array(data)
    
    data //= 255
    #for i in data:
     #   cv2.imshow('ANTESDATA= ', i*255)
      #  cv2.waitKey(0)
   # cv2.imshow('DATA= ', data[6])
    data = torch.from_numpy(data)
    #print("TEST= ",data)
    
    #R = data[:,0,:,:]
    #R[R==0.0] = 1.0
    #R[R==1.0] = 0.0
    
    
    return data             


if __name__ == "__main__":
    
    if torch.cuda.is_available():
        CUDA_ = True
    else:
        CUDA_ = False
         
    # CUDA_ = False
    
    
    N, inData = inputLoader()
    outData = outputLoader()
    #print("OUTDATA== ", outData)
    
    data = list(zip(inData, outData)) 
    #c,v = zip(*data)
    #print("V==",v)

    #batch =1
    #trainLoader = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False, num_workers=6)
    #print("trainLoader= ", trainLoader)
    print("N", N)
       
    num_classes = 2
    net = model = UNet(num_classes, depth=6, merge_mode='concat')
    iouSoma = 0.0
    #model.load_state_dict(torch.load("UNET1.pt"))
    
    model.load_state_dict(torch.load("UNET200epochs6Deep.pt"))
    
    if CUDA_:
        model.cuda()
    
    for i, (image, labels) in enumerate(data):
        print("LABELSS== ", labels)
        #print("i", i+1)
        labels3=labels.cpu().data.numpy()
       # cv2.imshow("labels", labels3*255)
    
        print("LABELS: ", labels.shape)
        print("imageBefore: ", image.shape)
        
        image = image.unsqueeze(0) # output format in columns
        print("imageAfter: ", image.shape)
        image = image.float()
        print(image.shape)
     
        #cv2.imshow("img", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        inImg = image.cpu().data.numpy();
        
        
        print(inImg.shape) #(1, 3, 320, 320)
        
        inImg = inImg.swapaxes(1,3)
        #print('2=',inImg.shape) #(1, 320, 320, 3)
        inImg = inImg.swapaxes(1,2)
        
        #print("3=",inImg.shape)#(1, 320, 320, 3)
        
        inImg = (inImg)*0.37+0.94
        
        inImg[0,:,:,:] = cv2.cvtColor(inImg[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imshow("in", inImg[0,:,:,:])
        
        cv2.waitKey(0)
              
        net.eval()
        
        if CUDA_:
            image = Variable(image.cuda())              
            labels = Variable(labels.cuda())
        else:
            image = Variable(image)
            labels = Variable(labels)
                               
             
        output = net(image)
        
        print("output shape=", output.shape)

        output = F.softmax(output, 1)
     #   out2  = netImage2image(output)
      
       
        print("L=== ", labels.shape)
        print("OOO=== ",output.shape) 
        out = output[0, 1, :, :].clone()
        
        #out2 = output[0, 1, :, :] + output[0, 0, :, :]
        
        #print("out2=", out2)
        print("OUTBEFORE = ", out)
        out[out>0.6] = 1.0
        out[out<=0.6] = 0.0
        print("out ", out)
        labels2 = labels.clone().float()
        out2 = out.clone()
       
        #print("LABELS2 ", labels2)
        #print("labels2*out2= ", torch.sum(labels2*out2))
        #print("labels2+out2= ", torch.sum(labels2+out2))
        intersection = torch.sum(labels2*out2)
        union = torch.sum(labels2+out2)         
        iou = intersection/ (union-intersection)
        iouSoma = iouSoma + iou.float()
        print("SOM= ", iouSoma)
        print("IOU= ", iou)
        print("TYPE== ", labels)
        labelsEnd = labels.cpu().data.numpy().copy()
        
        print("out3 ", labelsEnd.shape)
        out__ = out.cpu().data.numpy().copy()
        print("OUT_=== ",out__.shape) 
        cv2.imwrite("Validation.png",labelsEnd)
        cv2.imshow("Real", labels.cpu().data.numpy().copy()*255)
        cv2.imshow("out", out__)
       
        cv2.waitKey()
      
         
                
        
   

