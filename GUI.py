from PyQt5 import QtWidgets
import sys


def run1():
    print ('Run1')
    w.close()
    
def run2():
    print ('Run2')
    w.close()    
    
#INITIALIZE   
app = QtWidgets.QApplication(sys.argv)
w = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()

#BUILD WINDOW STRUCTURE
w.setGeometry(50, 50, 500, 300)
w.setWindowTitle("Emotional Neural Network")

#CREATE ALL LAYOUT COMPONENTS AND ACTION
#line1_edit = QtWidgets.QLineEdit()
#line2_edit = QtWidgets.QLineEdit()
btn_runTest = QtWidgets.QPushButton("Run1")
#btn_runTest.resize(50,50)
btn_runTest.move(100,100)
btn_runTest.clicked.connect(run1)
btn_runTrain = QtWidgets.QPushButton("Run2")
#btn_runTrain.resize(50,100)
btn_runTest.move(100,200)
btn_runTrain.clicked.connect(run2)

#BUILD LAYOUT: build it and put all the components
#layout.addWidget(line1_edit)
#layout.addWidget(line2_edit)
layout.addWidget(btn_runTest)
layout.addWidget(btn_runTrain)

#ASSIGN WINDOW LAYOUT
w.setLayout(layout)
w.show()

sys.exit(app.exec_())

