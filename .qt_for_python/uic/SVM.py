# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\aaa\OneDrive - The Northcap university\Desktop\UTS\UTS-Project\ui_files\SVM.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.columns = QtWidgets.QListWidget(self.centralwidget)
        self.columns.setGeometry(QtCore.QRect(10, 60, 391, 181))
        self.columns.setObjectName("columns")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(430, 320, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(430, 350, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(430, 380, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(430, 410, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(430, 440, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(430, 470, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(430, 500, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.c_ = QtWidgets.QLineEdit(self.centralwidget)
        self.c_.setGeometry(QtCore.QRect(620, 320, 111, 23))
        self.c_.setObjectName("c_")
        self.degree = QtWidgets.QLineEdit(self.centralwidget)
        self.degree.setGeometry(QtCore.QRect(620, 380, 111, 23))
        self.degree.setObjectName("degree")
        self.coef = QtWidgets.QLineEdit(self.centralwidget)
        self.coef.setGeometry(QtCore.QRect(620, 440, 111, 23))
        self.coef.setObjectName("coef")
        self.max_iter = QtWidgets.QLineEdit(self.centralwidget)
        self.max_iter.setGeometry(QtCore.QRect(620, 470, 111, 23))
        self.max_iter.setObjectName("max_iter")
        self.train = QtWidgets.QPushButton(self.centralwidget)
        self.train.setGeometry(QtCore.QRect(500, 540, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train.setFont(font)
        self.train.setObjectName("train")
        self.test_data = QtWidgets.QLineEdit(self.centralwidget)
        self.test_data.setGeometry(QtCore.QRect(900, 60, 71, 23))
        self.test_data.setObjectName("test_data")
        self.test_size_btn = QtWidgets.QPushButton(self.centralwidget)
        self.test_size_btn.setGeometry(QtCore.QRect(900, 100, 71, 23))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.test_size_btn.setFont(font)
        self.test_size_btn.setObjectName("test_size_btn")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(760, 60, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(760, 260, 20, 241))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.kernel = QtWidgets.QComboBox(self.centralwidget)
        self.kernel.setGeometry(QtCore.QRect(620, 350, 111, 23))
        self.kernel.setObjectName("kernel")
        self.kernel.addItem("")
        self.kernel.addItem("")
        self.kernel.addItem("")
        self.kernel.addItem("")
        self.kernel.addItem("")
        self.gamma = QtWidgets.QComboBox(self.centralwidget)
        self.gamma.setGeometry(QtCore.QRect(620, 410, 111, 23))
        self.gamma.setObjectName("gamma")
        self.gamma.addItem("")
        self.gamma.addItem("")
        self.gamma.addItem("")
        self.dec_func = QtWidgets.QComboBox(self.centralwidget)
        self.dec_func.setGeometry(QtCore.QRect(620, 500, 111, 23))
        self.dec_func.setObjectName("dec_func")
        self.dec_func.addItem("")
        self.dec_func.addItem("")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(430, 110, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.rmse = QtWidgets.QLabel(self.centralwidget)
        self.rmse.setGeometry(QtCore.QRect(620, 170, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.rmse.setFont(font)
        self.rmse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.rmse.setText("")
        self.rmse.setObjectName("rmse")
        self.mse = QtWidgets.QLabel(self.centralwidget)
        self.mse.setGeometry(QtCore.QRect(620, 140, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mse.setFont(font)
        self.mse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mse.setText("")
        self.mse.setObjectName("mse")
        self.mae = QtWidgets.QLabel(self.centralwidget)
        self.mae.setGeometry(QtCore.QRect(620, 110, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mae.setFont(font)
        self.mae.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mae.setText("")
        self.mae.setObjectName("mae")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(430, 170, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(430, 140, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.conf_mat = QtWidgets.QPushButton(self.centralwidget)
        self.conf_mat.setGeometry(QtCore.QRect(790, 330, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.conf_mat.setFont(font)
        self.conf_mat.setObjectName("conf_mat")
        self.dwnld = QtWidgets.QPushButton(self.centralwidget)
        self.dwnld.setGeometry(QtCore.QRect(650, 200, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.dwnld.setFont(font)
        self.dwnld.setObjectName("dwnld")
        self.report = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.report.setGeometry(QtCore.QRect(10, 290, 391, 171))
        self.report.setObjectName("report")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(0, 0, 1001, 51))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(22)
        self.label_19.setFont(font)
        self.label_19.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_19.setObjectName("label_19")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(10, 251, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_9.setFont(font)
        self.label_9.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);\n"
"")
        self.label_9.setObjectName("label_9")
        self.accuracy = QtWidgets.QLabel(self.centralwidget)
        self.accuracy.setGeometry(QtCore.QRect(580, 60, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.accuracy.setFont(font)
        self.accuracy.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.accuracy.setText("")
        self.accuracy.setObjectName("accuracy")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(440, 60, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(420, 250, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_10.setFont(font)
        self.label_10.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(430, 290, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.randomstate = QtWidgets.QLineEdit(self.centralwidget)
        self.randomstate.setGeometry(QtCore.QRect(620, 290, 111, 23))
        self.randomstate.setObjectName("randomstate")
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        self.predict.setGeometry(QtCore.QRect(440, 630, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.predict.setFont(font)
        self.predict.setObjectName("predict")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(20, 590, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.predict_val = QtWidgets.QLabel(self.centralwidget)
        self.predict_val.setGeometry(QtCore.QRect(770, 520, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.predict_val.setFont(font)
        self.predict_val.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.predict_val.setStatusTip("")
        self.predict_val.setAutoFillBackground(False)
        self.predict_val.setStyleSheet("color: rgb(83, 255, 71);\n"
"background-color: rgb(255, 255, 255);")
        self.predict_val.setObjectName("predict_val")
        self.ExitButton = QtWidgets.QPushButton(self.centralwidget)
        self.ExitButton.setGeometry(QtCore.QRect(890, 630, 91, 31))
        font = QtGui.QFont()
        font.setFamily("Segoe Fluent Icons")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.ExitButton.setFont(font)
        self.ExitButton.setStyleSheet("\n"
"background-color: rgb(255, 0, 0);\n"
"color: rgb(255, 255, 255);\n"
"")
        self.ExitButton.setObjectName("ExitButton")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(790, 470, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.label_17 = QtWidgets.QLabel(self.centralwidget)
        self.label_17.setGeometry(QtCore.QRect(800, 250, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_17.setFont(font)
        self.label_17.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_17.setObjectName("label_17")
        self.list = QtWidgets.QLineEdit(self.centralwidget)
        self.list.setGeometry(QtCore.QRect(230, 590, 541, 31))
        self.list.setObjectName("list")
        self.plot = QtWidgets.QPushButton(self.centralwidget)
        self.plot.setGeometry(QtCore.QRect(790, 380, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.plot.setFont(font)
        self.plot.setObjectName("plot")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "C:"))
        self.label_3.setText(_translate("MainWindow", "Kernel:"))
        self.label_4.setText(_translate("MainWindow", "Degree:"))
        self.label_5.setText(_translate("MainWindow", "Gamma:"))
        self.label_6.setText(_translate("MainWindow", "Coef:"))
        self.label_7.setText(_translate("MainWindow", "Max_Iter:"))
        self.label_8.setText(_translate("MainWindow", "Decision_func_shape:"))
        self.c_.setText(_translate("MainWindow", "1.0"))
        self.degree.setText(_translate("MainWindow", "3"))
        self.coef.setText(_translate("MainWindow", "0.0"))
        self.max_iter.setText(_translate("MainWindow", "-1"))
        self.train.setText(_translate("MainWindow", "Train"))
        self.test_data.setText(_translate("MainWindow", "0.1"))
        self.test_size_btn.setText(_translate("MainWindow", "Set"))
        self.label_12.setText(_translate("MainWindow", "Test Data Size"))
        self.kernel.setItemText(0, _translate("MainWindow", "rbf"))
        self.kernel.setItemText(1, _translate("MainWindow", "linear"))
        self.kernel.setItemText(2, _translate("MainWindow", "poly"))
        self.kernel.setItemText(3, _translate("MainWindow", "sigmoid"))
        self.kernel.setItemText(4, _translate("MainWindow", "precomputed"))
        self.gamma.setItemText(0, _translate("MainWindow", "scale"))
        self.gamma.setItemText(1, _translate("MainWindow", "auto"))
        self.gamma.setItemText(2, _translate("MainWindow", "custom"))
        self.dec_func.setItemText(0, _translate("MainWindow", "ovr"))
        self.dec_func.setItemText(1, _translate("MainWindow", "ovo"))
        self.label_13.setText(_translate("MainWindow", "Mean Absolute Error:"))
        self.label_14.setText(_translate("MainWindow", "Root Mean Sq. Error:"))
        self.label_15.setText(_translate("MainWindow", "Mean Square Error:"))
        self.conf_mat.setText(_translate("MainWindow", "Confusion Matrix"))
        self.dwnld.setText(_translate("MainWindow", "Download"))
        self.label_19.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">SVM Model</p></body></html>"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Classification Report:</p><p align=\"center\"><br/></p><p align=\"center\"><br/></p></body></html>"))
        self.label_24.setText(_translate("MainWindow", "Accuracy Score:"))
        self.label_10.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Hyper Parameters</p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "Random State:"))
        self.randomstate.setText(_translate("MainWindow", "144"))
        self.predict.setText(_translate("MainWindow", "Predict"))
        self.label_16.setText(_translate("MainWindow", "Enter List for Prediction :"))
        self.predict_val.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.ExitButton.setText(_translate("MainWindow", "Exit"))
        self.label_22.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Predicted Value</span></p></body></html>"))
        self.label_17.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Visualization</p><p align=\"center\"><br/></p></body></html>"))
        self.plot.setText(_translate("MainWindow", "Decission Regions"))
