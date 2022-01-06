#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Final project developed by Henrique das Virgens

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from collections import namedtuple
import numpy as np

import cv2

import glob

from UNET import UNet


# Method for loading training data
def inputLoader():
    
    data = []
    N=0
    for fn in sorted(glob.glob('dataIn/*.jpg', recursive=False)): # The glob module finds the file directory. If the recursion is true it will correspond to any directory/subdirectories, symbolic links to the directory
    ###for fn in glob.glob('/mnt/xfs/nucleus/nucleusTrain/**/*.png', recursive=True):
             if N<10:
               print(fn)
       ### if ('images' in fn):
             img = cv2.imread(fn) # reading color images (320, 480, 3)
             #print("Forma= ", img.shape)
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting from BGR to RGB
            
             img = cv2.resize(img, (320,320) ) # formato (320, 320, 3)
             
             img = img.astype(float) # convert image file to float64 matrix
             
             img = img/255.0
             
             img = (img-0.94)/0.37
             
             #print("input loader=", img.shape)
            
             
             img2 = img.copy()
             #print("swapaxes=", img2.shape)
             img_ = img.swapaxes(0,2) # swap it's like a transposition from (320, 320, 3) to (3320, 320)
             #print("input loader1=", img_.shape)
             img_ = img_.swapaxes(1,2)
             
             #print("input loader2=", img_.shape)
             #print("N= ", N)
            
             data.append(img_)
            # print("img2=", img2.shape)
             f1 = cv2.flip(img2, 0) #(320,320,3) Flip reverses the order of elements in a matrix, ie the matrix shape is kept but changes the order of the elements
             #print("flip=", f1.shape)
             f1 = f1.swapaxes(0,2) #(3,320,320)
             #print("flip2=", f1.shape)
             f1 = f1.swapaxes(1,2)
             #print("flip3=", f1.shape)
             data.append(f1)
    

             f2 = cv2.flip(img2, 1)
             f2 = f2.swapaxes(0,2)
             f2 = f2.swapaxes(1,2)
             data.append(f2)  

             f3 = cv2.flip(img2, -1)
             f3 = f3.swapaxes(0,2)
             f3 = f3.swapaxes(1,2)
             
             data.append(f3)                         , 
             
             
             
             N = N + 1
   
          
    data = np.array(data)  # Creating an array (object)
    #print("Data1= ",data.dtype) #float64
    data = torch.from_numpy(data) # Putting in the torch memory
    #print("Data2= ",data.dtype) #torch.float64
    
     
    
    return N*4, data          
             



def outputLoader():
    
    data = []
    N=0
    for fn in sorted(glob.glob('dataOut/*.jpg', recursive=False)):
             
    ###for fn in glob.glob('/mnt/xfs/nucleus/nucleusTrain/**/*.webp', recursive=True):
             
             if N<10:
                print(fn)
             N = N + 1   
       ### if ('images' in fn):
             img = cv2.imread(fn)
             
             img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting BGR to gray
             #cv2.imshow("image",img)
             #cv2.waitKey(0)
             
             img = cv2.resize(img, (320,320) )
            
             img[img>=1] = 255
             img[img<1] = 0
             
             img2 = img.copy()
             #img = img.swapaxes(0, 1)
             
             
             data.append(img)
             
             f1 = cv2.flip(img2, 0)
             #f1 = f1.swapaxes(0, 1)
             
             data.append(f1)
             
             f2 = cv2.flip(img2, 1)
             #f2 = f2.swapaxes(0, 1)
             
             data.append(f2)
             
             f3 = cv2.flip(img2, -1)
             #f3 = f3.swapaxes(0, 1)
             
             data.append(f3)
             
             
             
             
    
    data = np.array(data)
    
    data //= 255
    
    data = torch.from_numpy(data)
    
    #R = data[:,0,:,:]
    #R[R==0.0] = 1.0
    #R[R==1.0] = 0.0
    
    
    return data             

def netImage2image(outputs):
      outputsSoft = F.softmax(outputs, 1) # Generating a vector that represents a list with possible results
      o = outputsSoft.permute(2, 3, 0, 1) # permute() is used to swap the axes, ie the data is moved
      print("O", o.shape)
      outputsSoft = o.contiguous().view(-1, num_classes) # view reshapes the tensor to a different but compatible shape
      print("OUTSOFT", outputsSoft.shape)
      newOut = outputsSoft[:,1]
      print("NEWOUT = ", newOut)
      newOut[newOut > 0.5] = 1
      newOut[newOut <= 0.5] = 0
      
      #max_value = torch.max(newOut).item()
      #print("Acuracia:  ", max_value)
      
      return newOut    

if __name__ == "__main__":
    
     
    N, inData = inputLoader()
    outData = outputLoader()
    
   
    print("N=", N) # 472
    
    print(np.version.version)
    
    data = list(zip(inData, outData))
    
    
    batch = 6   # defines the number of samples that will be propagated over the network
    trainLoader = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=False, num_workers=6)

    
    num_classes = 2
    net = model = UNet(num_classes, depth=6, merge_mode='concat')
    # model.load_state_dict(torch.load("UNET1.pt"))
    
    print('run model...')
    
    
    """For the algorithm to run on a GPU the CUDA method in the model is called. the excerpt
     code below tells Pytorch to run the code on the GPU:
    """
    if torch.cuda.is_available():
        model.cuda()
        
    """Cross-Entropy Loss: Calculates the loss of the classification network by predicting the
     possibilities, which should be summed up to 1, as the softmax layer. the cross
     entropy loss increases when the predicted possibility diverges from the correct possibility.
    """
    criterion = nn.CrossEntropyLoss()
    
    #criterion = nn.BCELoss()
    
    """ Explain Adam Optimization
     Using the Adam model to do the optimization, that is, an equation for
     update the parameters of a model.
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, betas=(0.99, 0.9999))
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=0.000001)
    
    print('start train')
    
    
    bestAccuracy = 0.0
    
    iou200 = 0.0
    iouBest = 0.0
    cc = 0
    num_epochs = 50
    best = 1000000
    avgLoss = 0.0
    iter = 0
    old_epoch = 0
    for epoch in range(num_epochs):
        print("epoch= ", epoch)
        
        for i, (images, labels) in enumerate(trainLoader):
                
                print('iter=', i)
                print('epoch=', epoch)
                
                images = images.float() #(15, 3, 320, 320)
                labels = labels.long()  #(15, 320, 320)
                
            
                #print(images[0])
                print('images=', images.shape)
                print('label=', labels.shape)
                
                #print(labels[0])
                
              
                img2 =  images[0].clone()
                
                img2 = (images[0])*0.37+0.94
                
                img2 = img2.cpu().data.numpy();
                img2 = img2.swapaxes(0,2) # works as a transposition
                img2 = img2.swapaxes(0,1)
                
                img2[:,:,:] = cv2.cvtColor(img2[:,:,:], cv2.COLOR_RGB2BGR)
                
                #cv2.imshow("in", img2[:,:,:])
                
                #cv2.waitKey(0)
                
                
                img3 = labels[0].cpu().data.numpy().copy();
                
             #   print("max=", np.amax(img3))
              #  print("min=", np.amin(img3))
               # print("img3= ", img3.shape)
                
                #img3 = img3.swapaxes(0,1)
                                
                #img3[:,:] = cv2.cvtColor(img3[:,:], cv2.COLOR_RGB2BGR)
                
                img3 = img3.astype(np.uint8)

                #cv2.imshow("out", img3*255)
               # cv2.waitKey() 
                #cv2.destroyAllWindows()
                
                net.train()
                if torch.cuda.is_available():
                    images = Variable(images.cuda())
                    labels = Variable(labels.cuda())
                else:
                    images = Variable(images)
                    labels = Variable(labels)
              
                
              
                # set the gradient to zero before doing backwards (backpropagation)
                optimizer.zero_grad()
                
                outputs = net(images)   # passing the input batch to the template
                
                ##out__ = outputs.cpu().data.numpy()
                print("out=  ", outputs.shape) #([15,2, 320, 320])
                print("labels", labels.shape)  #([15, 320, 320])
                #labels2 = labels.clone().float()
                out2  = netImage2image(outputs)
                print("OUT2== ", out2)
                oo = outputs.permute(2, 3, 0, 1) # generating permutations in the list (distinct change of order of elements)
                la = labels.permute(1, 2, 0)
                
                
                
                print("oo= ", oo.shape)
                print("la= ", la.shape)
                
                outputs = oo.contiguous().view(-1, num_classes) # rows are stored as continuous blocks of memory, num_classes = number of columns
                labels = la.contiguous().view(-1) # the view has the function of reshaping the tensor, -1 because it doesn't know the number of lines
                
                
                print("out=  ", outputs.shape)
                print("labels", labels.shape)
                
    
                labels2 = labels.clone().int()
                intersection = torch.sum(out2*labels2)  # Will be zero if Truth=0 or Prediction=0
                union = torch.sum(out2+labels2)
                print("OUT2SHAPE= ", out2.shape)
                if union > 0:
                    iou = intersection/ (union- intersection)
                
                    print("IOU= ", iou)
                    
                 
                
                
                loss = criterion(outputs, labels) # getting the loss of negative log probability between our network and our target batch data
                
                loss.backward() #Backpropagation
                    
                # Updating parameters
                optimizer.step()
                
                
                print('loss=', loss.data)
               
 #               print("LABEL", labels)
  #              print("OUTPUTS", outputs)
                
                avgLoss +=  loss.data
                
                if avgLoss < best and old_epoch != epoch:
                  best = avgLoss
                  old_epoch = epoch             
                  iouBest = iou
                  torch.save(model.state_dict(), "UNETbatch5deep6.pt")
                  if (epoch==200):
                        #torch.save(model.state_dict(), "UNET200epochs6Deep.pt")
                        iou200 = iou
                  
                    
                    
        old_epoch = epoch
        avgLoss = 0.0
        print("IOU 200 = ",iou200)        
        print("BEST IOU= ",iouBest)
                
                
                
            
            
   # x = Variable(torch.FloatTensor(np.random.random((1, 3, 320, 320))))
   # out = model(x)
    
   # print(out)
    
   # outimg = out.view(320,320,3).data.numpy()
    
   # outimg *= 255
    
   # outimg = outimg.astype(np.uint8)
    
   # print(outimg.shape)
    
   # out = out.permute(2, 3, 0, 1).contiguous()#.view(-1, 3)
    #loss = torch.sum(out)
    #loss.backward()
    

    #print(out)
    
    
    
   # cv2.imshow('img', outimg )
    
  #  cv2.waitKey()
    
    
    
    
