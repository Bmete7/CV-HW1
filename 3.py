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
 
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
 
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
 
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()
 
 
    def plot(self):
        data = [random.random() for i in range(25)]
        ax = self.figure.add_subplot(111)
        ax.plot(data, 'r-')
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
        #self.lab1 = QLabel()
        self.vBox1 = QVBoxLayout()
        #qp1 = QPixmap('color2.png')
        #self.lab1.setPixmap(qp1)
        #self.vBox1.addWidget(self.lab1)
        groupBox1.setLayout(self.vBox1)
        
        groupBox2 = QGroupBox('Target File')
        #self.lab2 = QLabel()
        self.vBox2 = QVBoxLayout()
#        qp2 = QPixmap('color2.png')
#        self.lab2.setPixmap(qp2)
#        vBox2.addWidget(self.lab2)
        groupBox2.setLayout(self.vBox2)
        
        groupBox3 = QGroupBox('Output File')
#        self.lab3 = QLabel()
        self.vBox3 = QVBoxLayout()
#        qp3 = QPixmap('color2.png')
#        self.lab3.setPixmap(qp3)
#            vBox3.addWidget(self.lab3)
        groupBox3.setLayout(self.vBox3)
        
        hBox = QHBoxLayout()
        
        hBox.addWidget(groupBox1)
        hBox.addWidget(groupBox2)
        hBox.addWidget(groupBox3)
        

        self.setLayout(hBox)
        self.setGeometry(0, 0, 0,0)
        
    
    def targetImage(self,fN):
    
        lab= QLabel()
        qp = QPixmap(fN)
        lab.setPixmap(qp)
        self.vBox2.addWidget(lab)
        
        
#        qp2 = QPixmap(fN)
#        self.lab1.setPixmap(qp2)
#    
    def inputImage(self,fN):
        lab= QLabel()
        qp = QPixmap(fN)
        lab.setPixmap(qp)
        self.vBox1.addWidget(lab)
        
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = "Histogram Matching"
        self.top = 50
        self.left = 50
        self.width = 1200
        self.height = 800
        self.image = None
        self.inputFile = ''
        self.targetFile= ''
        self.initWindow()
        
    def initWindow(self):
         
        exitAct = QAction(QIcon('exit.png'), '&Exit' , self)
        importAct = QAction('&Open Input' , self)
        targetAct = QAction('&Open Target' , self)
        
        exitAct.setShortcut('Ctrl+Q')
        
        exitAct.setStatusTip('Exit application')
        importAct.setStatusTip('Open Input')
        targetAct.setStatusTip('Open Target')
        
        exitAct.triggered.connect(self.closeApp)
        importAct.triggered.connect(self.importImage)
        targetAct.triggered.connect(self.importTarget)
        
        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        
        fileMenu.addAction(exitAct)
        fileMenu.addAction(importAct)
        fileMenu.addAction(targetAct)

        self.content = ExampleContent(self, 'color2.png', 'color1.png')
        self.setCentralWidget(self.content)
        
        #m = PlotCanvas(self, width=5, height=4)
        #m.move(500,500)
        
        self.setWindowTitle(self.title)
        self.setGeometry( self.top, self.left, self.width, self.height)
        self.show()

    
    def closeApp(self):
        sys.exit()
    
    def importTarget(self):
        fileName = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Png Files (*.png)")
        self.targetFile = fileName[0]
        image = cv2.imread(fileName[0])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.content.targetImage(self.targetFile)
    
    def importImage(self):
        fileName = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Png Files (*.png)")
        self.inputFile = fileName[0]
        image = cv2.imread(fileName[0])
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        self.content.inputImage(self.inputFile)

if __name__ == '__main__':
    App = QApplication(sys.argv)
    window = Window()
    cv2.destroyAllWindows()
    sys.exit(App.exec())
    