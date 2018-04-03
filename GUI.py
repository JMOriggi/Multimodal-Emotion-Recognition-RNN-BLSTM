import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QCheckBox, QWidget, QFileDialog, QPushButton
from PyQt5.QtCore import QSize    

#DEFINE GLOBAL VARIABLE ROOT 
MainRoot = ''
mainRootT = ''
     
#CLASS FOR WINDOW CREATION     
class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
 
        #WINDOW SIZE
        self.setGeometry(600, 200, 500, 300)
        #self.setMinimumSize(QSize(140, 40))    
        self.setWindowTitle("Emotional Neural Network") 
        
        #TRAINING1 CHECKBOX
        self.b1 = QCheckBox("ORIGINAL",self)
        self.b1.stateChanged.connect(self.clickBoxOriginal)
        self.b1.move(20,20)
        #TRAINING2 CHECKBOX
        self.b4 = QCheckBox("FAKE",self)
        self.b4.stateChanged.connect(self.clickBoxFake)
        self.b4.move(20,50)
        #TRAINING3 CHECKBOX
        self.b6 = QCheckBox("LAV",self)
        self.b6.stateChanged.connect(self.clickBoxLav)
        self.b6.move(20,80)
        
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

