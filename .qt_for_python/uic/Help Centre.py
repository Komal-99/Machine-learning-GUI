# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\aaa\OneDrive - The Northcap university\Desktop\UTS\UTS-Project\ui_files\Help Centre.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1408, 920)
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(0, 0, 1661, 931))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.widget.setFont(font)
        self.widget.setStyleSheet("background-color:rgb(255, 255, 255)")
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(0, 60, 1411, 61))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255)")
        self.label.setObjectName("label")
        self.toolButton_2 = QtWidgets.QToolButton(self.widget)
        self.toolButton_2.setGeometry(QtCore.QRect(0, 630, 1411, 51))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(11)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(10)
        self.toolButton_2.setFont(font)
        self.toolButton_2.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"font: 87 11pt \"Arial Black\";\n"
"color: rgb(255, 255, 255)")
        self.toolButton_2.setObjectName("toolButton_2")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setGeometry(QtCore.QRect(390, 720, 261, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe Fluent Icons")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("background-color: rgb(200, 217, 231);\n"
"border-radius: 35px;\n"
"border-width: 2opx;\n"
"color: rgb(1, 78, 144);")
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.widget)
        self.pushButton_2.setGeometry(QtCore.QRect(810, 720, 261, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe Fluent Icons")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setStyleSheet("background-color: rgb(200, 217, 231);\n"
"border-radius: 35px;\n"
"border-width: 2opx;\n"
"color: rgb(1, 78, 144);")
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(130, 190, 431, 61))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255)")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(820, 190, 411, 61))
        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(28)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255)")
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Help Centre</span></p></body></html>"))
        self.toolButton_2.setToolTip(_translate("Dialog", "<html><head/><body><p><span style=\" font-size:16pt;\">FAQ\'s</span></p></body></html>"))
        self.toolButton_2.setText(_translate("Dialog", "For more information visit www.uts.com"))
        self.pushButton.setText(_translate("Dialog", "BACK"))
        self.pushButton_2.setText(_translate("Dialog", "CONTINUE"))
        self.label_2.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Getting Started</span></p></body></html>"))
        self.label_3.setToolTip(_translate("Dialog", "<html><head/><body><p><br/></p></body></html>"))
        self.label_3.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt;\">Steps</span></p></body></html>"))
