# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:59:51 2018

@author: BurakBey
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
image = cv2.imread('C:/Users/BurakBey/Desktop/BLG 453E- Computer Vision/hw1/BLG453E_hw1/color2.png', 3)
image2 = cv2.imread('C:/Users/BurakBey/Desktop/BLG 453E- Computer Vision/hw1/BLG453E_hw1/color1.png', 3)
result = np.zeros((image.shape[0] ,  image.shape[1] , image.shape[2]), dtype = 'uint8' ) 
result2 = np.zeros((image.shape[0] ,  image.shape[1] , image.shape[2]), dtype = 'uint8' ) 

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

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

mi = np.min(image)
Mi = np.max(image)
mj = np.min(image2)
Mj = np.max(image2)
gj = mj



for gi in range(mi,Mi):
    while gj<256 and cdfInput[0,gi]<1 and cdfTarget[0,gj]<cdfInput[0,gi]:
        gj += 1
    result[image==gi]=gj
    time.sleep(0)

result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

aaa = cv2.equalizeHist(image[:,:,0])
aa=cv2.equalizeHist(image[:,:,1])
a= cv2.equalizeHist(image[:,:,2])
result2[:,:,0]= aaa
result2[:,:,1]= aa
result2[:,:,2]= a

cv2.imshow('re', result2)
cv2.waitKey(0)

bbb = cv2.equalizeHist(result[:,:,0])
bb=cv2.equalizeHist(result[:,:,1])
b = cv2.equalizeHist(result[:,:,2])
result2[:,:,0]= bbb
result2[:,:,1]= bb
result2[:,:,2]= b

cv2.imshow('rme', result2)
cv2.waitKey(0)

cv2.imshow('re', result)
cv2.imshow('input', image )
cv2.imshow('target', image2 )
cv2.waitKey(0)

