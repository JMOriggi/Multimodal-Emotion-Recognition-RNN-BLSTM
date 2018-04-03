from PyQt5 import QtWidgets
import sys


def main():
    print ('Starting Program')
    w.close()
    
#INITIALIZE APP    
app = QtWidgets.QApplication(sys.argv)

#INITIALIZE WINDOW
w = QtWidgets.QWidget()
w.setGeometry(50, 50, 500, 300)
w.setWindowTitle("Emotional Neural Network")


#CREATE ALL LAYOUT COMPONENTS AND ACTION
line1_edit = QtWidgets.QLineEdit()
line2_edit = QtWidgets.QLineEdit()
run_btn = QtWidgets.QPushButton("Run")
run_btn.clicked.connect(main)

#BUILD LAYOUT: build it and put all the components
layout = QtWidgets.QVBoxLayout()
layout.addWidget(line1_edit)
layout.addWidget(line2_edit)
layout.addWidget(run_btn)

#ASSIGN WINDOW LAYOUT
w.setLayout(layout)
w.show()

sys.exit(app.exec_())