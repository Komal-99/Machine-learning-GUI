# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\jvish\OneDrive\Desktop\UTS-Project\ui_files\LinearRegression.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LinearRegression(object):
    def setupUi(self, LinearRegression):
        LinearRegression.setObjectName("LinearRegression")
        LinearRegression.resize(1000, 600)
        self.centralwidget = QtWidgets.QWidget(LinearRegression)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1001, 51))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.columns = QtWidgets.QListWidget(self.centralwidget)
        self.columns.setGeometry(QtCore.QRect(20, 70, 431, 161))
        self.columns.setObjectName("columns")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(20, 350, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 380, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.fit_inter = QtWidgets.QComboBox(self.centralwidget)
        self.fit_inter.setGeometry(QtCore.QRect(160, 350, 121, 23))
        self.fit_inter.setObjectName("fit_inter")
        self.fit_inter.addItem("")
        self.fit_inter.addItem("")
        self.normalize = QtWidgets.QComboBox(self.centralwidget)
        self.normalize.setGeometry(QtCore.QRect(160, 380, 121, 23))
        self.normalize.setObjectName("normalize")
        self.normalize.addItem("")
        self.normalize.addItem("")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(310, 270, 20, 181))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(600, 150, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(610, 80, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.intercept = QtWidgets.QLabel(self.centralwidget)
        self.intercept.setGeometry(QtCore.QRect(730, 70, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.intercept.setFont(font)
        self.intercept.setStyleSheet("border-color: rgb(255, 255, 255);\n"
"background-color: rgb(255, 255, 255);")
        self.intercept.setText("")
        self.intercept.setObjectName("intercept")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 280, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.test_data = QtWidgets.QLineEdit(self.centralwidget)
        self.test_data.setGeometry(QtCore.QRect(160, 280, 71, 23))
        self.test_data.setObjectName("test_data")
        self.train = QtWidgets.QPushButton(self.centralwidget)
        self.train.setGeometry(QtCore.QRect(60, 410, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train.setFont(font)
        self.train.setObjectName("train")
        self.bar_plot = QtWidgets.QPushButton(self.centralwidget)
        self.bar_plot.setGeometry(QtCore.QRect(690, 260, 251, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.bar_plot.setFont(font)
        self.bar_plot.setObjectName("bar_plot")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(520, 230, 401, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(340, 320, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(340, 350, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(340, 380, 171, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.mse = QtWidgets.QLabel(self.centralwidget)
        self.mse.setGeometry(QtCore.QRect(530, 350, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mse.setFont(font)
        self.mse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mse.setText("")
        self.mse.setObjectName("mse")
        self.accuracy = QtWidgets.QLabel(self.centralwidget)
        self.accuracy.setGeometry(QtCore.QRect(480, 265, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.accuracy.setFont(font)
        self.accuracy.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.accuracy.setText("")
        self.accuracy.setObjectName("accuracy")
        self.rmse = QtWidgets.QLabel(self.centralwidget)
        self.rmse.setGeometry(QtCore.QRect(530, 380, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.rmse.setFont(font)
        self.rmse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.rmse.setText("")
        self.rmse.setObjectName("rmse")
        self.weights = QtWidgets.QTextBrowser(self.centralwidget)
        self.weights.setGeometry(QtCore.QRect(730, 130, 231, 81))
        self.weights.setObjectName("weights")
        self.dwnld = QtWidgets.QPushButton(self.centralwidget)
        self.dwnld.setGeometry(QtCore.QRect(470, 410, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.dwnld.setFont(font)
        self.dwnld.setObjectName("dwnld")
        self.mae = QtWidgets.QLabel(self.centralwidget)
        self.mae.setGeometry(QtCore.QRect(530, 320, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mae.setFont(font)
        self.mae.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mae.setText("")
        self.mae.setObjectName("mae")
        self.train_2 = QtWidgets.QPushButton(self.centralwidget)
        self.train_2.setGeometry(QtCore.QRect(410, 500, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train_2.setFont(font)
        self.train_2.setObjectName("train_2")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(20, 460, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(720, 410, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.predict_val = QtWidgets.QLabel(self.centralwidget)
        self.predict_val.setGeometry(QtCore.QRect(720, 440, 271, 51))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.predict_val.setFont(font)
        self.predict_val.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.predict_val.setStatusTip("")
        self.predict_val.setAutoFillBackground(False)
        self.predict_val.setStyleSheet("color:rgb(1, 78, 144);\n"
"background-color: rgb(255, 255, 255);")
        self.predict_val.setObjectName("predict_val")
        self.ExitButton = QtWidgets.QPushButton(self.centralwidget)
        self.ExitButton.setGeometry(QtCore.QRect(870, 510, 91, 31))
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
        self.test_size_btn = QtWidgets.QPushButton(self.centralwidget)
        self.test_size_btn.setGeometry(QtCore.QRect(240, 280, 71, 23))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.test_size_btn.setFont(font)
        self.test_size_btn.setObjectName("test_size_btn")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(390, 260, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.list = QtWidgets.QLineEdit(self.centralwidget)
        self.list.setGeometry(QtCore.QRect(230, 460, 471, 31))
        self.list.setObjectName("list")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(10, 240, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_8.setObjectName("label_8")
        self.error = QtWidgets.QLabel(self.centralwidget)
        self.error.setGeometry(QtCore.QRect(230, 500, 141, 16))
        self.error.setStyleSheet("color: rgb(255, 0, 4);")
        self.error.setText("")
        self.error.setObjectName("error")
        self.split = QtWidgets.QLabel(self.centralwidget)
        self.split.setGeometry(QtCore.QRect(100, 310, 171, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.split.setFont(font)
        self.split.setStyleSheet("color: rgb(1, 78, 144);")
        self.split.setText("")
        self.split.setObjectName("split")
        self.target = QtWidgets.QLabel(self.centralwidget)
        self.target.setGeometry(QtCore.QRect(840, 410, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.target.setFont(font)
        self.target.setObjectName("target")
        LinearRegression.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(LinearRegression)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 26))
        self.menubar.setObjectName("menubar")
        LinearRegression.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(LinearRegression)
        self.statusbar.setObjectName("statusbar")
        LinearRegression.setStatusBar(self.statusbar)

        self.retranslateUi(LinearRegression)
        QtCore.QMetaObject.connectSlotsByName(LinearRegression)

    def retranslateUi(self, LinearRegression):
        _translate = QtCore.QCoreApplication.translate
        LinearRegression.setWindowTitle(_translate("LinearRegression", "MainWindow"))
        self.label.setText(_translate("LinearRegression", "<html><head/><body><p align=\"center\">Linear Regression Model</p></body></html>"))
        self.label_3.setText(_translate("LinearRegression", "<html><head/><body><p>Fit_Intercept :</p></body></html>"))
        self.label_4.setText(_translate("LinearRegression", "<html><head/><body><p>Normalize :</p></body></html>"))
        self.fit_inter.setItemText(0, _translate("LinearRegression", "True"))
        self.fit_inter.setItemText(1, _translate("LinearRegression", "False"))
        self.normalize.setItemText(0, _translate("LinearRegression", "False"))
        self.normalize.setItemText(1, _translate("LinearRegression", "True"))
        self.label_5.setText(_translate("LinearRegression", "Coef (Weights) : "))
        self.label_6.setText(_translate("LinearRegression", "Intercept : "))
        self.label_7.setText(_translate("LinearRegression", "<html><head/><body><p>Test Train Split</p></body></html>"))
        self.test_data.setText(_translate("LinearRegression", "0.1"))
        self.train.setText(_translate("LinearRegression", "Train"))
        self.bar_plot.setText(_translate("LinearRegression", "Predicted vs Actual Bar PLot"))
        self.label_11.setText(_translate("LinearRegression", "Mean Absolute Error:"))
        self.label_12.setText(_translate("LinearRegression", "Mean Square Error:"))
        self.label_13.setText(_translate("LinearRegression", "Root Mean Sq. Error:"))
        self.dwnld.setText(_translate("LinearRegression", "Download Model"))
        self.train_2.setText(_translate("LinearRegression", "Predict"))
        self.label_14.setText(_translate("LinearRegression", "Enter List for Prediction :"))
        self.label_15.setText(_translate("LinearRegression", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Predicted </span></p></body></html>"))
        self.predict_val.setText(_translate("LinearRegression", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.ExitButton.setText(_translate("LinearRegression", "Exit"))
        self.test_size_btn.setText(_translate("LinearRegression", "Set"))
        self.label_16.setText(_translate("LinearRegression", "Accuracy"))
        self.list.setText(_translate("LinearRegression", "0.1"))
        self.label_8.setText(_translate("LinearRegression", "<html><head/><body><p align=\"center\">Hyper Parameters</p></body></html>"))
        self.target.setText(_translate("LinearRegression", "<html><head/><body><p><br/></p></body></html>"))
