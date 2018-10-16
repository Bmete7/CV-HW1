# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:59:51 2018

@author: BurakBey
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('C:/Users/BurakBey/Desktop/BLG 453E- Computer Vision/hw1/BLG453E_hw1/color1.png', 3)

image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


imageRedChannel = image[:,:,0]
imageGreenChannel = image[:,:,1]
imageBlueChannel = image[:,:,2]



imageBlueChannel = np.reshape(imageBlueChannel,((imageBlueChannel.shape[0]*imageBlueChannel.shape[1]),1) )
imageGreenChannel = np.reshape(imageGreenChannel,((imageGreenChannel.shape[0]*imageGreenChannel.shape[1]),1) )
imageRedChannel = np.reshape(imageRedChannel,((imageRedChannel.shape[0]*imageRedChannel.shape[1]),1) )





def findHistograms(image):
    counts = np.zeros((256,1), dtype='int32')
    for i in range(image.shape[0]):
        counts[image[i]] += 1
    
    print([counts[1]])
    return counts

countGreen = np.zeros((256,1), dtype='int32')
countRed = np.zeros((256,1), dtype='int32')
countBlue = np.zeros((256,1), dtype='int32')

countGreen= findHistograms(imageGreenChannel)
countBlue= findHistograms(imageBlueChannel)
countRed= findHistograms(imageRedChannel)

plt.bar(np.arange(256), height = countGreen , color = 'green')
plt.show()
plt.bar(np.arange(256), height = countRed , color = 'red')
plt.show()
plt.bar(np.arange(256), height = countBlue , color = 'blue')
plt.show()



#cv2.imshow('NEW WNDOW' , image)
#cv2.waitKey(0)