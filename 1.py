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

resultRed = np.zeros((image.shape[0] ,  image.shape[1]), dtype = 'uint8' ) 
resultGreen = np.zeros((image.shape[0] ,  image.shape[1]), dtype = 'uint8' ) 
resultBlue = np.zeros((image.shape[0] ,  image.shape[1]), dtype = 'uint8' )  

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

imageRedChannel = image[:,:,0]
imageGreenChannel = image[:,:,1]
imageBlueChannel = image[:,:,2]

imageRedRaw = image[:,:,0]
imageGreenRaw = image[:,:,1]
imageBlueRaw = image[:,:,2]

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
    while gj<256 and cdfInput[0,gi]<1 and cdfTarget[0,gj]<cdfInput[0,gi] :
        gj += 1
    resultRed[imageRedRaw==gi ] =gj
gj = mj
for gi in range(mi,Mi):
    while gj<256 and cdfInput[1,gi]<1 and cdfTarget[1,gj]<cdfInput[1,gi] :
        gj += 1
    resultGreen[imageGreenRaw==gi ] =gj
gj = mj    
for gi in range(mi,Mi):
    while gj<256 and cdfInput[2,gi]<1 and cdfTarget[2,gj]<cdfInput[2,gi] :
        gj += 1
    resultBlue[imageBlueRaw==gi ] =gj

result[:,:,0] = resultBlue
result[:,:,1] = resultGreen
result[:,:,2] = resultRed

RedChannel = cv2.equalizeHist(result[:,:,0])
GreenChannel = cv2.equalizeHist(result[:,:,1])
BlueChannel = cv2.equalizeHist(result[:,:,2])
BlueChannel = np.reshape(BlueChannel,((BlueChannel.shape[0]*BlueChannel.shape[1]),1) )
GreenChannel = np.reshape(GreenChannel,((GreenChannel.shape[0]*GreenChannel.shape[1]),1) )
RedChannel = np.reshape(RedChannel,((RedChannel.shape[0]*RedChannel.shape[1]),1) )


countResult = np.zeros((3,256,1), dtype='int32')
countResult[0]= findHistograms(RedChannel)
countResult[1] = findHistograms(GreenChannel)
countResult[2] = findHistograms(BlueChannel)


pdfResult = np.zeros((3,256,1), dtype = 'float64')
cdfResult = np.zeros((3,256,1), dtype = 'float64')
plt.bar(np.arange(256), height = countResult[1] , color = 'green')
plt.show()
plt.close()
plt.bar(np.arange(256), height = countResult[2] , color = 'blue')
plt.show()
plt.close()
plt.bar(np.arange(256), height = countResult[0] , color = 'red')
plt.show()
plt.close()
k= 0 
for i in range(3):
    k = 0
    for j in countResult[i]:
        pdfResult[i,k] = j/BlueChannel.size
        k+=1

for m in range(3):
    for i in range(256):
        if(i == 0):
            cdfResult[m,i] = pdfResult[m,i]
        else:
            cdfResult[m,i] = cdfResult[m,i-1] + pdfResult[m,i]



resultEqualize = np.zeros((image.shape[0] ,  image.shape[1] , image.shape[2]), dtype = 'uint8' ) 

for i in range(3):
    for j in range(result.shape[0]):
        for k in range(result.shape[1]):
            resultEqualize[j,k,i] = 255* cdfResult[i,result[j,k,i]]
        

resultEqualize = cv2.cvtColor(resultEqualize, cv2.COLOR_RGB2BGR)
cv2.imshow('assd', resultEqualize)
result= cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

cv2.imshow('assda', result)
cv2.waitKey(0)



result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)

cv2.imshow('asd', result)
cv2.waitKey(0)