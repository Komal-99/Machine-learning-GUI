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
        LinearRegression.resize(1116, 710)
        self.centralwidget = QtWidgets.QWidget(LinearRegression)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.columns = QtWidgets.QListWidget(self.centralwidget)
        self.columns.setObjectName("columns")
        self.gridLayout.addWidget(self.columns, 1, 0, 2, 6)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 1, 8, 1, 3)
        self.intercept = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.intercept.setFont(font)
        self.intercept.setStyleSheet("border-color: rgb(255, 255, 255);\n"
"background-color: rgb(255, 255, 255);")
        self.intercept.setText("")
        self.intercept.setObjectName("intercept")
        self.gridLayout.addWidget(self.intercept, 1, 11, 1, 2)
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 3, 0, 2, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 8, 1, 2)
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 4, 5, 2, 1)
        self.weights = QtWidgets.QTextBrowser(self.centralwidget)
        self.weights.setObjectName("weights")
        self.gridLayout.addWidget(self.weights, 2, 11, 1, 2)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 3, 7, 1, 6)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 5, 3, 7, 1)
        self.accuracy = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.accuracy.setFont(font)
        self.accuracy.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.accuracy.setText("")
        self.accuracy.setObjectName("accuracy")
        self.gridLayout.addWidget(self.accuracy, 4, 6, 3, 3)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 6, 0, 1, 1)
        self.test_data = QtWidgets.QLineEdit(self.centralwidget)
        self.test_data.setObjectName("test_data")
        self.gridLayout.addWidget(self.test_data, 6, 1, 1, 1)
        self.bar_plot = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.bar_plot.setFont(font)
        self.bar_plot.setObjectName("bar_plot")
        self.gridLayout.addWidget(self.bar_plot, 4, 9, 3, 4)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 13)
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 9, 4, 1, 3)
        self.rmse = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.rmse.setFont(font)
        self.rmse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.rmse.setText("")
        self.rmse.setObjectName("rmse")
        self.gridLayout.addWidget(self.rmse, 9, 7, 1, 5)
        self.mse = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mse.setFont(font)
        self.mse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mse.setText("")
        self.mse.setObjectName("mse")
        self.gridLayout.addWidget(self.mse, 8, 7, 1, 5)
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 7, 4, 1, 3)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 8, 4, 1, 3)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 9, 0, 1, 1)
        self.normalize = QtWidgets.QComboBox(self.centralwidget)
        self.normalize.setObjectName("normalize")
        self.normalize.addItem("")
        self.normalize.addItem("")
        self.gridLayout.addWidget(self.normalize, 9, 1, 1, 2)
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.gridLayout.addWidget(self.label_15, 10, 10, 1, 2)
        self.target = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.target.setFont(font)
        self.target.setObjectName("target")
        self.gridLayout.addWidget(self.target, 10, 12, 1, 1)
        self.split = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.split.setFont(font)
        self.split.setStyleSheet("color: rgb(1, 78, 144);")
        self.split.setText("")
        self.split.setObjectName("split")
        self.gridLayout.addWidget(self.split, 7, 0, 1, 3)
        self.mae = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mae.setFont(font)
        self.mae.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mae.setText("")
        self.mae.setObjectName("mae")
        self.gridLayout.addWidget(self.mae, 7, 7, 1, 5)
        self.test_size_btn = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.test_size_btn.setFont(font)
        self.test_size_btn.setObjectName("test_size_btn")
        self.gridLayout.addWidget(self.test_size_btn, 6, 2, 1, 1)
        self.fit_inter = QtWidgets.QComboBox(self.centralwidget)
        self.fit_inter.setObjectName("fit_inter")
        self.fit_inter.addItem("")
        self.fit_inter.addItem("")
        self.gridLayout.addWidget(self.fit_inter, 8, 1, 1, 2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 8, 0, 1, 1)
        self.ExitButton = QtWidgets.QPushButton(self.centralwidget)
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
        self.gridLayout.addWidget(self.ExitButton, 13, 12, 1, 1)
        self.error = QtWidgets.QLabel(self.centralwidget)
        self.error.setStyleSheet("color: rgb(255, 0, 4);")
        self.error.setText("")
        self.error.setObjectName("error")
        self.gridLayout.addWidget(self.error, 13, 2, 1, 3)
        self.train = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train.setFont(font)
        self.train.setObjectName("train")
        self.gridLayout.addWidget(self.train, 10, 0, 2, 2)
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 12, 0, 1, 2)
        self.predict_val = QtWidgets.QLabel(self.centralwidget)
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
        self.gridLayout.addWidget(self.predict_val, 11, 10, 2, 3)
        self.list = QtWidgets.QLineEdit(self.centralwidget)
        self.list.setObjectName("list")
        self.gridLayout.addWidget(self.list, 12, 2, 1, 8)
        self.dwnld = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.dwnld.setFont(font)
        self.dwnld.setObjectName("dwnld")
        self.gridLayout.addWidget(self.dwnld, 10, 6, 2, 3)
        self.train_2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train_2.setFont(font)
        self.train_2.setObjectName("train_2")
        self.gridLayout.addWidget(self.train_2, 13, 5, 1, 3)
        LinearRegression.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(LinearRegression)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1116, 26))
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
        self.label_6.setText(_translate("LinearRegression", "Intercept : "))
        self.label_8.setText(_translate("LinearRegression", "<html><head/><body><p align=\"center\">Hyper Parameters</p></body></html>"))
        self.label_5.setText(_translate("LinearRegression", "Coef (Weights) : "))
        self.label_16.setText(_translate("LinearRegression", "Accuracy"))
        self.label_7.setText(_translate("LinearRegression", "<html><head/><body><p>Test Train Split</p></body></html>"))
        self.test_data.setText(_translate("LinearRegression", "0.1"))
        self.bar_plot.setText(_translate("LinearRegression", "Predicted vs Actual Bar PLot"))
        self.label.setText(_translate("LinearRegression", "<html><head/><body><p align=\"center\">Linear Regression Model</p></body></html>"))
        self.label_13.setText(_translate("LinearRegression", "Root Mean Sq. Error:"))
        self.label_11.setText(_translate("LinearRegression", "Mean Absolute Error:"))
        self.label_12.setText(_translate("LinearRegression", "Mean Square Error:"))
        self.label_4.setText(_translate("LinearRegression", "<html><head/><body><p>Normalize :</p></body></html>"))
        self.normalize.setItemText(0, _translate("LinearRegression", "False"))
        self.normalize.setItemText(1, _translate("LinearRegression", "True"))
        self.label_15.setText(_translate("LinearRegression", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Predicted </span></p></body></html>"))
        self.target.setText(_translate("LinearRegression", "<html><head/><body><p><span style=\" font-weight:600;\"><br/></span></p></body></html>"))
        self.test_size_btn.setText(_translate("LinearRegression", "Set"))
        self.fit_inter.setItemText(0, _translate("LinearRegression", "True"))
        self.fit_inter.setItemText(1, _translate("LinearRegression", "False"))
        self.label_3.setText(_translate("LinearRegression", "<html><head/><body><p>Fit_Intercept :</p></body></html>"))
        self.ExitButton.setText(_translate("LinearRegression", "Exit"))
        self.train.setText(_translate("LinearRegression", "Train"))
        self.label_14.setText(_translate("LinearRegression", "Enter List for Prediction :"))
        self.predict_val.setText(_translate("LinearRegression", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.list.setText(_translate("LinearRegression", "0.1"))
        self.dwnld.setText(_translate("LinearRegression", "Download Model"))
        self.train_2.setText(_translate("LinearRegression", "Predict"))
