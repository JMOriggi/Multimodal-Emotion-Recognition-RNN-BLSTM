import sys
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QCheckBox, QWidget, QFileDialog, QPushButton, QTextEdit
from PyQt5.QtCore import QSize 
from PyQt5.QtGui import QFont, QColor 
import MainSetCorpus as SetCorp  
import MainTraining as train
import MainPredict as predict


     
#CLASS FOR WINDOW CREATION     
class Window(QMainWindow):
    
    #DEFINE GLOBAL VARIABLE ROOT 
    MainRoot = ''
    MainRootT = ''
    AudioTextFlag = 0 #2 for both, 1 for only text, 0 for only audio
    FlagLM = 0 #0 create new model file during training, 1 use the one already existing
    
    def __init__(self):
        QMainWindow.__init__(self)
 
        #WINDOW SIZE
        self.setGeometry(600, 200, 740, 600) 
        self.setWindowTitle("Emotional Neural Network") 
        
        # Set window background color
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), QColor(255, 239, 201))
        self.setPalette(p)
        
        #CHECK POINTS GRAFIC
        x = 20
        ya = 20
        yb = 120
        yc = 220
        xbutton = 520
        ybutton = 30
        
        #LABELS
        fontLabel = QFont()
        fontLabel.setBold(True)
        fontLabel.setPixelSize(13)
        self.l1 = QLabel(self)
        self.l1.setText('Select Corpus')
        self.l1.setFont(fontLabel)
        self.l1.move(x,ya)
        self.l2 = QLabel(self)
        self.l2.setText('Select NN type')
        self.l2.setFont(fontLabel)
        self.l2.move(x,yb)
        self.l3 = QLabel(self)
        self.l3.setText('Load Existing Models')
        self.l3.setFont(fontLabel)
        self.l3.setFixedSize(150,20)
        self.l3.move(x,yc)
        
        #CHECKBOXS
        fontBox = QFont()
        fontBox.setPixelSize(12)
        self.b1 = QCheckBox("Original Corpus",self)
        self.b1.stateChanged.connect(self.clickBoxOriginal)
        self.b1.setFont(fontBox)
        self.b1.move(x+10,ya+20)
        self.b2 = QCheckBox("Fake Corpus",self)
        self.b2.stateChanged.connect(self.clickBoxFake)
        self.b2.setFont(fontBox)
        self.b2.move(x+10,ya+40)
        self.b3 = QCheckBox("Lav",self)
        self.b3.stateChanged.connect(self.clickBoxLav)
        self.b3.setFont(fontBox)
        self.b3.move(x+10,ya+60)
        self.b4 = QCheckBox("Audio",self)
        self.b4.stateChanged.connect(self.clickBoxA)
        self.b4.setFont(fontBox)
        self.b4.move(x+10,yb+20)
        self.b5 = QCheckBox("Text",self)
        self.b5.stateChanged.connect(self.clickBoxT)
        self.b5.setFont(fontBox)
        self.b5.move(x+10,yb+40)
        self.b6 = QCheckBox("Audio+Text",self)
        self.b6.stateChanged.connect(self.clickBoxAT)
        self.b6.setFont(fontBox)
        self.b6.move(x+10,yb+60)
        self.b7 = QCheckBox("Audio Model",self)
        self.b7.stateChanged.connect(self.clickBoxFlagMA)
        self.b7.setFont(fontBox)
        self.b7.move(x+10,yc+20)
        self.b8 = QCheckBox("Text Model",self)
        self.b8.stateChanged.connect(self.clickBoxFlagMT)
        self.b8.setFont(fontBox)
        self.b8.move(x+10,yc+40)
        
        #BUTTONS
        self.btn_setCorpus = QPushButton("SET CORPUS", self)
        self.btn_setCorpus.clicked.connect(self.runSetCorpus)
        self.btn_setCorpus.setFixedSize(100,50)
        self.btn_setCorpus.move(xbutton,ybutton)
        self.btn_runTraining = QPushButton("RUN TRAINING", self)
        self.btn_runTraining.clicked.connect(self.runTraining)
        self.btn_runTraining.setFixedSize(100,50)
        self.btn_runTraining.move(xbutton,ybutton+60)
        self.btn_runTest = QPushButton("RUN TEST", self)
        self.btn_runTest.clicked.connect(self.runTest)
        self.btn_runTest.setFixedSize(100,50)
        self.btn_runTest.move(xbutton,ybutton+120)
        
        #CONSOLE LOG
        self.logOutput = QTextEdit(self)
        self.logOutput.setReadOnly(True)
        self.logOutput.setLineWrapMode(QTextEdit.NoWrap)
        font = self.logOutput.font()
        font.setFamily("Courier")
        font.setPointSize(14)
        self.logOutput.setFixedSize(700,250)
        self.logOutput.move(20,300)
        #Button clear log
        self.btn_clear = QPushButton("Clear Logs", self)
        self.btn_clear.clicked.connect(self.clearLog)
        self.btn_clear.setFixedSize(100,25)
        self.btn_clear.move(620,560)   
    
    #PRINT LOG
    def printLog(self, text):
        self.logOutput.insertPlainText(text+'\n')
        sb = self.logOutput.verticalScrollBar()
        sb.setValue(sb.maximum()) 
    def clearLog(self):
        self.logOutput.clear()    
        
    #CHECKBOX METHODS          
    def clickBoxOriginal(self, state):
        if state == QtCore.Qt.Checked:
            self.MainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Training')
            self.MainRootT = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test')
            txt = 'Checked on roots: '+self.MainRoot+', AND, '+self.MainRootT
            self.printLog(txt)
        else:
            self.MainRoot = ''
            self.printLog('Unchecked clickBoxOriginal') 
    def clickBoxFake(self, state):
        if state == QtCore.Qt.Checked:
            self.MainRoot = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
            self.MainRootT = os.path.normpath('D:\DATA\POLIMI\----TESI-----\Corpus_Test_Training')
            txt = 'Checked on roots: '+self.MainRoot+', AND, '+self.MainRootT
            self.printLog(txt)
        else:
            self.MainRoot = ''
            self.printLog('Unchecked clickBoxFake') 
    def clickBoxLav(self, state):
        if state == QtCore.Qt.Checked:
            self.MainRoot = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
            self.MainRootT = os.path.normpath(r'C:\Users\JORIGGI00\Documents\MyDOCs\Corpus_Test_Training')
            txt = 'Checked on roots: '+self.MainRoot+', AND, '+self.MainRootT
            self.printLog(txt)
        else:
            self.MainRoot = ''
            self.printLog('Unchecked clickBoxLav')                
    def clickBoxA(self, state):
        if state == QtCore.Qt.Checked:
            self.AudioTextFlag = 0
            txt = 'Checked only audio'
            self.printLog(txt)
        else:
            self.AudioTextFlag = 0
    def clickBoxT(self, state):
        if state == QtCore.Qt.Checked:
            self.AudioTextFlag = 1
            txt = 'Checked only text'
            self.printLog(txt)
        else:
            self.AudioTextFlag = 0  
    def clickBoxAT(self, state):
        if state == QtCore.Qt.Checked:
            self.AudioTextFlag = 2
            txt = 'Checked audio and text'
            self.printLog(txt)
        else:
            self.AudioTextFlag = 0               
    def clickBoxFlagMA(self, state):
        if state == QtCore.Qt.Checked:
            self.FlagLMA = 1
            txt = 'Checked Load Model Audio'
            self.printLog(txt)
        else:
            self.FlagLMA = 0 
    def clickBoxFlagMT(self, state):
        if state == QtCore.Qt.Checked:
            self.FlagLMT = 1
            txt = 'Checked Load Model Text'
            self.printLog(txt)
        else:
            self.FlagLMT = 0               
    
    #BUTTONS METHODS 
    def runSetCorpus(self, w):
        if self.MainRoot == '':
            self.printLog('Select one Root checkbox')
        else:
            txt = 'Set Corpus on roots: '+self.MainRoot+', AND, '+self.MainRootT
            self.printLog(txt)
            SetCorp.mainSetCorpus(self.MainRoot,self.MainRootT)
            self.printLog('END of set Corpus')
            
    def runTraining(self, w):
        if self.MainRoot == '':
            self.printLog('Select one Root checkbox')
        else:
            txt = 'Run training with root: '+self.MainRoot
            self.printLog(txt) 
            train.mainTraining(self.MainRoot, self.AudioTextFlag, self.FlagLMA, self.FlagLMT)
            self.printLog('END of Training') 
            
    def runTest(self, w):
        if self.MainRoot == '':
            self.printLog('Select one Root checkbox')
        else:
            txt = 'Set Test with roots: '+self.MainRoot+', AND, '+self.MainRootT
            self.printLog(txt) 
            predict.mainPredict(self.MainRoot, self.MainRootT, self.AudioTextFlag)
            self.printLog('END of Test')     
            
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Window()
    mainWin.show()
    sys.exit(app.exec_())

