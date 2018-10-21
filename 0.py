# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:28:02 2018

@author: BurakBey
"""

import cv2
 
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import random

class PlotCanvas(FigureCanvas):
 
    
    
    def __init__(self, val , order , parent=None, width=5, height=4, dpi=100):
        self.val = None
        self.val = val
        self.countReds = None
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot(self.val,order)
 

        
    def plot(self,val,order):
        color = ''
        if(order == 0):
            color = 'red'
        elif(order == 1):
            color = 'green'
        else:
            color = 'blue'
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.bar(np.arange(256), height = val[:,:,order], color =  color)
        ax.set_title('PyQt Matplotlib Example')
        self.draw()
    

class ExampleContent(QWidget):
    def __init__(self, parent,fileName,fileName2):
        self.fileName = fileName
        self.fileName2 = fileName2
        self.parent = parent
        
        
        QWidget.__init__(self, parent)
        self.initUI(fileName,fileName2)


        
    def initUI(self,fileName,fileName2):        

        groupBox1 = QGroupBox('Input File')
        self.vBox1 = QVBoxLayout()
        groupBox1.setLayout(self.vBox1)
        
        groupBox2 = QGroupBox('Target File')
        self.vBox2 = QVBoxLayout()
        groupBox2.setLayout(self.vBox2)
        
        groupBox3 = QGroupBox('Output File')
        self.vBox3 = QVBoxLayout()
        groupBox3.setLayout(self.vBox3)
        hBox = QHBoxLayout()
        
        hBox.addWidget(groupBox1)
        hBox.addWidget(groupBox2)
        hBox.addWidget(groupBox3)
        

        self.setLayout(hBox)
        self.setGeometry(0, 0, 0,0)
        
    
    def targetImage(self,fN,val):    
        lab= QLabel()
        qp = QPixmap(fN)
        lab.setPixmap(qp)
        
        mRed= PlotCanvas(val,0,self)
        mGreen = PlotCanvas(val,1,self)
        mBlue= PlotCanvas(val,2,self)
        
        mRed.setFixedSize(400,200)
        mGreen.setFixedSize(400,200)
        mBlue.setFixedSize(400,200)
        
        self.vBox2.addWidget(lab)
        self.vBox2.addWidget(mRed)
        self.vBox2.addWidget(mGreen)
        self.vBox2.addWidget(mBlue)
        
        
    def inputImage(self,fN,val):
        lab= QLabel()
        qp = QPixmap(fN)
        lab.setPixmap(qp)
        
        self.vBox1.addWidget(lab)

        mRed= PlotCanvas(val,0,self)
        mGreen = PlotCanvas(val,1,self)
        mBlue= PlotCanvas(val,2,self)
        
        mRed.setFixedSize(400,200)
        mGreen.setFixedSize(400,200)
        mBlue.setFixedSize(400,200)
        
        self.vBox1.addWidget(mRed)
        self.vBox1.addWidget(mGreen)
        self.vBox1.addWidget(mBlue)
        
    def resultImage(self,fN,val):
        lab= QLabel()
        qp = QPixmap(fN)
        lab.setPixmap(qp)
        
        self.vBox3.addWidget(lab)
        
        mRed= PlotCanvas(val,0,self)
        mGreen = PlotCanvas(val,1,self)
        mBlue= PlotCanvas(val,2,self)
        
        mRed.setFixedSize(400,200)
        mGreen.setFixedSize(400,200)
        mBlue.setFixedSize(400,200)
        
        self.vBox3.addWidget(mRed)
        self.vBox3.addWidget(mGreen)
        self.vBox3.addWidget(mBlue)
        
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = "Histogram Matching"
        self.top = 50
        self.left = 50
        self.width = 1800
        self.height = 1200
        self.inImage = None
        self.tarImage = None
        self.result = None
        self.inputFile = ''
        self.targetFile= ''
        self.initWindow()
        self.inputFilled = False
        self.targetFilled = False
        self.InputChannels = np.zeros((256,1,3), dtype='int32')
        self.TargetChannels = np.zeros((256,1,3), dtype='int32')
        self.ResultChannels = np.zeros((256,1,3), dtype='int32')
        
    def initWindow(self):
         
        exitAct = QAction(QIcon('exit.png'), '&Exit' , self)
        importAct = QAction('&Open Input' , self)
        targetAct = QAction('&Open Target' , self)
        eqHistogram = QAction('&Equalize Histogram' , self)
    
        exitAct.setShortcut('Ctrl+Q')
        
        exitAct.setStatusTip('Exit application')
        importAct.setStatusTip('Open Input')
        targetAct.setStatusTip('Open Target')
        
        exitAct.triggered.connect(self.closeApp)
        importAct.triggered.connect(self.importInput)
        targetAct.triggered.connect(self.importTarget)
        eqHistogram.triggered.connect(self.createResultImage)
        
        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        
        fileMenu.addAction(exitAct)
        fileMenu.addAction(importAct)
        fileMenu.addAction(targetAct)

        self.content = ExampleContent(self, 'color2.png', 'color1.png')
        self.setCentralWidget(self.content)
        
        self.toolbar = self.addToolBar('Equalize Histogram')
        self.toolbar.addAction(eqHistogram)
        
        self.setWindowTitle(self.title)
        self.setStyleSheet('QMainWindow{background-color: darkgray;border: 1px solid black;}')
        self.setGeometry( self.top, self.left, self.width, self.height)
        self.show()

    
    def closeApp(self):
        sys.exit()
    
    
    def importInput(self):
        fileName = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Png Files (*.png)")
        self.inputFile = fileName[0]
        self.inImage = cv2.imread(fileName[0])
        self.inImage = cv2.cvtColor(self.inImage,cv2.COLOR_BGR2RGB)
        
        self.calculateHistogram(self.inImage,1)
        self.content.inputImage(self.inputFile,self.InputChannels)
        
        
        
    def importTarget(self):
        fileName = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Png Files (*.png)")
        self.targetFile = fileName[0]
        self.tarImage = cv2.imread(fileName[0])
        self.tarImage = cv2.cvtColor(self.tarImage,cv2.COLOR_BGR2RGB)
        
        self.calculateHistogram(self.tarImage,2)
        self.content.targetImage(self.targetFile,self.TargetChannels)
        
            
    def calculateHistogram(self,image,imType):
        imageRedChannel = image[:,:,0]
        imageGreenChannel = image[:,:,1]
        imageBlueChannel = image[:,:,2]
        imageBlueChannel = np.reshape(imageBlueChannel,((imageBlueChannel.shape[0]*imageBlueChannel.shape[1]),1) )
        imageGreenChannel = np.reshape(imageGreenChannel,((imageGreenChannel.shape[0]*imageGreenChannel.shape[1]),1) )
        imageRedChannel = np.reshape(imageRedChannel,((imageRedChannel.shape[0]*imageRedChannel.shape[1]),1) )
        countGreen = np.zeros((256,1), dtype='int32')
        countRed = np.zeros((256,1), dtype='int32')
        countBlue = np.zeros((256,1), dtype='int32')
        
        countGreen= self.findHistograms(imageGreenChannel)
        countBlue= self.findHistograms(imageBlueChannel)
        countRed= self.findHistograms(imageRedChannel)
        
    
        if(imType==1):
            self.InputChannels[:,:,0] = countRed
            self.InputChannels[:,:,1] = countGreen
            self.InputChannels[:,:,2] = countBlue
        elif(imType==2):
            self.TargetChannels[:,:,0] = countRed
            self.TargetChannels[:,:,1] = countGreen
            self.TargetChannels[:,:,2] = countBlue
        else:
            self.ResultChannels[:,:,0] = countRed
            self.ResultChannels[:,:,1] = countGreen
            self.ResultChannels[:,:,2] = countBlue
         
        
    def findHistograms(self,image):
        counts = np.zeros((256,1), dtype='int32')
        for i in range(image.shape[0]):
            counts[image[i]] += 1
            
        return counts
    
    def createResultImage(self):
        
        self.result = np.zeros((self.inImage.shape[0] ,  self.inImage.shape[1] , self.inImage.shape[2]), dtype = 'uint8' ) 

        resultRed = np.zeros((self.inImage.shape[0] ,  self.inImage.shape[1]), dtype = 'uint8' ) 
        resultGreen = np.zeros((self.inImage.shape[0] ,  self.inImage.shape[1]), dtype = 'uint8' ) 
        resultBlue = np.zeros((self.inImage.shape[0] ,  self.inImage.shape[1]), dtype = 'uint8' )  
        
        imageRedChannel = self.inImage[:,:,0]
        imageGreenChannel = self.inImage[:,:,1]
        imageBlueChannel = self.inImage[:,:,2]
        
        imageRedRaw = self.inImage[:,:,0]
        imageGreenRaw = self.inImage[:,:,1]
        imageBlueRaw = self.inImage[:,:,2]
        
        imageRedChannel2 = self.tarImage[:,:,0]
        imageGreenChannel2 = self.tarImage[:,:,1]
        imageBlueChannel2 = self.tarImage[:,:,2]
        
        
        imageBlueChannel = np.reshape(imageBlueChannel,((imageBlueChannel.shape[0]*imageBlueChannel.shape[1]),1) )
        imageGreenChannel = np.reshape(imageGreenChannel,((imageGreenChannel.shape[0]*imageGreenChannel.shape[1]),1) )
        imageRedChannel = np.reshape(imageRedChannel,((imageRedChannel.shape[0]*imageRedChannel.shape[1]),1) )
        
        
        imageBlueChannel2 = np.reshape(imageBlueChannel2,((imageBlueChannel2.shape[0]*imageBlueChannel2.shape[1]),1) )
        imageGreenChannel2 = np.reshape(imageGreenChannel2,((imageGreenChannel2.shape[0]*imageGreenChannel2.shape[1]),1) )
        imageRedChannel2 = np.reshape(imageRedChannel2,((imageRedChannel2.shape[0]*imageRedChannel2.shape[1]),1) )

        
        countInput = np.zeros((3,256,1), dtype='int32')
        countInput[0]= self.findHistograms(imageRedChannel)
        countInput[1] = self.findHistograms(imageGreenChannel)
        countInput[2] = self.findHistograms(imageBlueChannel)
        
        
        countTarget= np.zeros((3,256,1), dtype='int32')
        countTarget[0]= self.findHistograms(imageRedChannel2)
        countTarget[1] = self.findHistograms(imageGreenChannel2)
        countTarget[2] = self.findHistograms(imageBlueChannel2)
        
        
        
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
        
        mi = np.min(self.inImage)
        Mi = np.max(self.inImage)
        mj = np.min(self.tarImage)
        Mj = np.max(self.tarImage)
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
        
        self.result[:,:,2] = resultBlue
        self.result[:,:,1] = resultGreen
        self.result[:,:,0] = resultRed
        
        self.result= cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR) 
        cv2.imwrite('result.png', self.result)
        self.result= cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR) 
        self.calculateHistogram(self.result,3)
        self.content.resultImage('result.png',self.ResultChannels)
    
    
    
if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    cv2.destroyAllWindows()
    sys.exit(App.exec())
    