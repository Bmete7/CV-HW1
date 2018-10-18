# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:59:51 2018

@author: BurakBey
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/BurakBey/Desktop/BLG 453E- Computer Vision/hw1/BLG453E_hw1/color1.png', 3)
image2 = cv2.imread('C:/Users/BurakBey/Desktop/BLG 453E- Computer Vision/hw1/BLG453E_hw1/color2.png', 3)
result = np.zeros((image.shape[0] ,  image.shape[1] , image.shape[2]), dtype = 'int32' )

result[0] = cv2.equalizeHist(image[0],image2[0])
result[1] = cv2.equalizeHist(image[1],image2[1])
result[2] = cv2.equalizeHist(image[2],image2[2])
cv2.imshow('empty4', result)
cv2.waitKey(0)

image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2= cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)


imageRedChannel = image[:,:,0]
imageGreenChannel = image[:,:,1]
imageBlueChannel = image[:,:,2]

imageRedChannel2 = image2[:,:,0]
imageGreenChannel2 = image2[:,:,1]
imageBlueChannel2 = image2[:,:,2]


imageBlueChannel = np.reshape(imageBlueChannel,((imageBlueChannel.shape[0]*imageBlueChannel.shape[1]),1) )
imageGreenChannel = np.reshape(imageGreenChannel,((imageGreenChannel.shape[0]*imageGreenChannel.shape[1]),1) )
imageRedChannel = np.reshape(imageRedChannel,((imageRedChannel.shape[0]*imageRedChannel.shape[1]),1) )


imageBlueChannel2 = np.reshape(imageBlueChannel2,((imageBlueChannel2.shape[0]*imageBlueChannel2.shape[1]),1) )
imageGreenChannel2 = np.reshape(imageGreenChannel2,((imageGreenChannel2.shape[0]*imageGreenChannel2.shape[1]),1) )
imageRedChannel2 = np.reshape(imageRedChannel2,((imageRedChannel2.shape[0]*imageRedChannel2.shape[1]),1) )





def findHistograms(image):
    counts = np.zeros((256,1), dtype='int32')
    for i in range(image.shape[0]):
        counts[image[i]] += 1
        
    return counts


countInput = np.zeros((3,256,1), dtype='int32')
countInput[0]= findHistograms(imageRedChannel)
countInput[1] = findHistograms(imageGreenChannel)
countInput[2] = findHistograms(imageBlueChannel)


countTarget= np.zeros((3,256,1), dtype='int32')
countTarget[0]= findHistograms(imageRedChannel2)
countTarget[1] = findHistograms(imageGreenChannel2)
countTarget[2] = findHistograms(imageBlueChannel2)



pdfInput = np.zeros((3,256,1), dtype = 'float64')
cdfInput = np.zeros((3,256,1), dtype = 'float64')

pdfTarget = np.zeros((3,256,1), dtype = 'float64')
cdfTarget = np.zeros((3,256,1), dtype = 'float64')

k= 0 
for i in range(3):
    k = 0
    for j in countInput[i]:
        pdfInput[i,k] = j/imageBlueChannel.size
        k+=1

for m in range(3):
    for i in range(256):
        if(i == 0):
            cdfInput[m,i] = pdfInput[m,i]
        else:
            cdfInput[m,i] = cdfInput[m,i-1] + pdfInput[m,i]


k= 0 
for i in range(3):
    k = 0
    for j in countTarget[i]:
        pdfTarget[i,k] = j/imageBlueChannel.size
        k+=1

for m in range(3):
    for i in range(256):
        if(i == 0):
            cdfTarget[m,i] = pdfTarget[m,i]
        else:
            cdfTarget[m,i] = cdfTarget[m,i-1] + pdfTarget[m,i]

    
plt.plot(np.arange(256), cdfTarget[0] , color = 'red')
plt.plot(np.arange(256), cdfTarget[1] , color = 'green')
plt.plot(np.arange(256), cdfTarget[2] , color = 'blue')
    
plt.plot(np.arange(256), pdfTarget[0] , color = 'red')
plt.plot(np.arange(256), pdfTarget[1] , color = 'green')
plt.plot(np.arange(256), pdfTarget[2] , color = 'blue')



for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            result[i,j,k] = 255* cdfInput[k,image2[i,j,k]]
            
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imshow('empty', result)
cv2.waitKey(0)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            result[i,j,k] = 255* cdfInput[k,image2[i,j,k]]
            
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imshow('empty2', result)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            result[i,j,k] = 255* cdfTarget[k,image[i,j,k]]
            
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imshow('empty3', result)


for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            result[i,j,k] = 255* cdfTarget[k,image2[i,j,k]]
            
            

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            manipulated = False
            for m in range(1, 256):
                if(cdfTarget[k,m] >= cdfInput[k,image[i,j,k]] and cdfTarget[k,m-1] < cdfInput[k,image[i,j,k]]):
                    result[i,j,k] = 255* cdfTarget[k,m]
                    manipulated = True
            if(manipulated == False):
                result[i,j,k] = 255* cdfTarget[k,0]
            
            
            
            
            
result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imshow('empty4', result)
cv2.waitKey(0)