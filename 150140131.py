# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 22:28:02 2018

@author: BurakBey
"""

import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys



class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = "Histogram Matching"
        self.top = 100
        self.left = 100
        self.width = 580
        self.height = 500
        self.initWindow()
        
    def initWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry( self.top, self.left, self.width, self.height)
        self.show()
        
    
    
    
App = QApplication(sys.argv)

window = Window()
sys.exit(App.exec())
