# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\jvish\OneDrive\Desktop\UTS-Project\ui_files\modeltype.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModal)
        MainWindow.resize(701, 605)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMaximumSize(QtCore.QSize(780, 16777215))
        MainWindow.setStyleSheet("background-color: rgb(1,78,144);\n"
"")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.trainedmodel = QtWidgets.QPushButton(self.centralwidget)
        self.trainedmodel.setGeometry(QtCore.QRect(220, 290, 261, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe Fluent Icons")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.trainedmodel.setFont(font)
        self.trainedmodel.setStyleSheet("background-color: rgb(200, 217, 231);\n"
"border-radius: 35px;\n"
"border-width: 2opx;\n"
"color: rgb(1, 78, 144);")
        self.trainedmodel.setObjectName("trainedmodel")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 70, 541, 81))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(36)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:rgb(255, 255, 255);\n"
"font: 36pt \"Times New Roman\";")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.newmodel = QtWidgets.QPushButton(self.centralwidget)
        self.newmodel.setGeometry(QtCore.QRect(220, 190, 261, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe Fluent Icons")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.newmodel.setFont(font)
        self.newmodel.setStyleSheet("background-color: rgb(200, 217, 231);\n"
"border-radius: 35px;\n"
"border-width: 2opx;\n"
"color: rgb(1, 78, 144);")
        self.newmodel.setObjectName("newmodel")
        self.ExitButton = QtWidgets.QPushButton(self.centralwidget)
        self.ExitButton.setGeometry(QtCore.QRect(220, 380, 261, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe Fluent Icons")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.ExitButton.setFont(font)
        self.ExitButton.setStyleSheet("background-color: rgb(200, 217, 231);\n"
"border-radius: 35px;\n"
"border-width: 2opx;\n"
"color: rgb(1, 78, 144);")
        self.ExitButton.setObjectName("ExitButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 701, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.trainedmodel.setText(_translate("MainWindow", "Trained Model"))
        self.label_2.setToolTip(_translate("MainWindow", "<html><head/><body><p align=\"justify\">Predictive Maintenance</p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:22pt;\">CHOOSE YOUR MODEL</span></p></body></html>"))
        self.newmodel.setText(_translate("MainWindow", "New Model"))
        self.ExitButton.setText(_translate("MainWindow", "Exit"))
