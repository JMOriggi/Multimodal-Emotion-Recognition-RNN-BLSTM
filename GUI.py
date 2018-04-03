import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QCheckBox, QWidget, QFileDialog, QPushButton
from PyQt5.QtCore import QSize    

#DEFINE GLOBAL VARIABLE ROOT 
MainRoot = ''
mainRootT = ''
AudioTextFlag = 0 #2 for both, 1 for only text, 0 for only audio
     
#CLASS FOR WINDOW CREATION     
class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
 
        #WINDOW SIZE
        self.setGeometry(600, 200, 500, 300)
        #self.setMinimumSize(QSize(140, 40))    
        self.setWindowTitle("Emotional Neural Network") 
        
        #TRAINING1 CHECKBOX
        self.b1 = QCheckBox("Original Corpus",self)
        self.b1.stateChanged.connect(self.clickBoxOriginal)
        self.b1.move(20,20)
        #TRAINING2 CHECKBOX
        self.b2 = QCheckBox("Fake Corpus",self)
        self.b2.stateChanged.connect(self.clickBoxFake)
        self.b2.move(20,50)
        #TRAINING3 CHECKBOX
        self.b3 = QCheckBox("Lav",self)
        self.b3.stateChanged.connect(self.clickBoxLav)
        self.b3.move(20,80)
        #AUDIO TRAINING
        self.b4 = QCheckBox("Audio",self)
        self.b4.stateChanged.connect(self.clickBoxA)
        self.b4.move(20,130)
        #TEXT TRAINING
        self.b5 = QCheckBox("Text",self)
        self.b5.stateChanged.connect(self.clickBoxT)
        self.b5.move(20,150)
        #AUDIO+TEXT TRAINING
        self.b6 = QCheckBox("Audio+Text",self)
        self.b6.stateChanged.connect(self.clickBoxAT)
        self.b6.move(20,170)
        
        #BUTTON SET CORPUS
        self.btn_runTest = QPushButton("SET CORPUS", self)
        self.btn_runTest.clicked.connect(self.runSetCorpus)
        self.btn_runTest.setFixedSize(100,50)
        self.btn_runTest.move(340,20)
        
        #BUTTON TRAINING
        self.btn_runTest = QPushButton("RUN TRAINING", self)
        self.btn_runTest.clicked.connect(self.runTraining)
        self.btn_runTest.setFixedSize(100,50)
        self.btn_runTest.move(340,100)
        
        #BUTTON TEST
        self.btn_runTest = QPushButton("RUN TEST", self)
        self.btn_runTest.clicked.connect(self.runTest)
        self.btn_runTest.setFixedSize(100,50)
        self.btn_runTest.move(340,180)
              
    def clickBoxOriginal(self, state):
        if state == QtCore.Qt.Checked:
            MainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Training')
            MainRootT = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test')
            print('Checked clickBoxOriginal')
        else:
            MainRoot = ''
            print('Unchecked clickBoxOriginal')
            
    def clickBoxFake(self, state):
        if state == QtCore.Qt.Checked:
            MainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
            MainRootT = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
            print('Checked clickBoxFake')
        else:
            MainRoot = ''
            print('Unchecked clickBoxFake')
    
    def clickBoxLav(self, state):
        if state == QtCore.Qt.Checked:
            MainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
            MainRootT = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
            print('Checked clickBoxLav')
        else:
            MainRoot = ''
            print('Unchecked clickBoxLav')                
    
    def clickBoxA(self, state):
        if state == QtCore.Qt.Checked:
            AudioTextFlag = 0
            print('Checked clickBoxAT: ',AudioTextFlag)
        else:
            AudioTextFlag = 0
            print('Checked clickBoxAT: ',AudioTextFlag)
            
    def clickBoxT(self, state):
        if state == QtCore.Qt.Checked:
            AudioTextFlag = 1
            print('Checked clickBoxAT: ',AudioTextFlag)
        else:
            AudioTextFlag = 0
            print('Checked clickBoxAT: ',AudioTextFlag)  
            
    def clickBoxAT(self, state):
        if state == QtCore.Qt.Checked:
            AudioTextFlag = 2
            print('Checked clickBoxAT: ',AudioTextFlag)
        else:
            AudioTextFlag = 0
            print('Checked clickBoxAT: ',AudioTextFlag)               
 
 
    def runSetCorpus(self, w):
        if MainRoot == '':
            print ('Select one Root checkbox')
        else:
            print ('runSetCorpus')
            
    def runTraining(self, w):
        if MainRoot == '':
            print ('Select one Root checkbox')
        else:
            print ('runTraining') 
        
    def runTest(self, w):
        if MainRoot == '':
            print ('Select one Root checkbox')
        else:
            print ('runTest')     

            
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Window()
    mainWin.show()
    sys.exit(app.exec_())

