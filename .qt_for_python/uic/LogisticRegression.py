# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\aaa\OneDrive - The Northcap university\Desktop\UTS\UTS-Project\ui_files\LogisticRegression.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 689)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.columns = QtWidgets.QListWidget(self.centralwidget)
        self.columns.setGeometry(QtCore.QRect(20, 70, 391, 171))
        self.columns.setObjectName("columns")
        self.train = QtWidgets.QPushButton(self.centralwidget)
        self.train.setGeometry(QtCore.QRect(500, 450, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train.setFont(font)
        self.train.setObjectName("train")
        self.max_iter = QtWidgets.QLineEdit(self.centralwidget)
        self.max_iter.setGeometry(QtCore.QRect(560, 380, 111, 23))
        self.max_iter.setObjectName("max_iter")
        self.conf_mat = QtWidgets.QPushButton(self.centralwidget)
        self.conf_mat.setGeometry(QtCore.QRect(740, 320, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.conf_mat.setFont(font)
        self.conf_mat.setObjectName("conf_mat")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(430, 380, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.output = QtWidgets.QPushButton(self.centralwidget)
        self.output.setGeometry(QtCore.QRect(740, 380, 181, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.output.setFont(font)
        self.output.setObjectName("output")
        self.solver = QtWidgets.QComboBox(self.centralwidget)
        self.solver.setGeometry(QtCore.QRect(560, 350, 111, 23))
        self.solver.setObjectName("solver")
        self.solver.addItem("")
        self.solver.addItem("")
        self.solver.addItem("")
        self.solver.addItem("")
        self.solver.addItem("")
        self.label_22 = QtWidgets.QLabel(self.centralwidget)
        self.label_22.setGeometry(QtCore.QRect(430, 350, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.multi_class = QtWidgets.QComboBox(self.centralwidget)
        self.multi_class.setGeometry(QtCore.QRect(560, 410, 111, 23))
        self.multi_class.setObjectName("multi_class")
        self.multi_class.addItem("")
        self.multi_class.addItem("")
        self.multi_class.addItem("")
        self.label_23 = QtWidgets.QLabel(self.centralwidget)
        self.label_23.setGeometry(QtCore.QRect(430, 410, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.report = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.report.setGeometry(QtCore.QRect(20, 290, 391, 181))
        self.report.setObjectName("report")
        self.mae_2 = QtWidgets.QLabel(self.centralwidget)
        self.mae_2.setGeometry(QtCore.QRect(620, 140, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mae_2.setFont(font)
        self.mae_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mae_2.setText("")
        self.mae_2.setObjectName("mae_2")
        self.mae = QtWidgets.QLabel(self.centralwidget)
        self.mae.setGeometry(QtCore.QRect(620, 110, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.mae.setFont(font)
        self.mae.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.mae.setText("")
        self.mae.setObjectName("mae")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(430, 140, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(430, 170, 171, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.rmse = QtWidgets.QLabel(self.centralwidget)
        self.rmse.setGeometry(QtCore.QRect(620, 170, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.rmse.setFont(font)
        self.rmse.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.rmse.setText("")
        self.rmse.setObjectName("rmse")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(430, 110, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setObjectName("label_15")
        self.accuracy_score = QtWidgets.QLabel(self.centralwidget)
        self.accuracy_score.setGeometry(QtCore.QRect(580, 60, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.accuracy_score.setFont(font)
        self.accuracy_score.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.accuracy_score.setText("")
        self.accuracy_score.setObjectName("accuracy_score")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(0, 0, 1001, 51))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(22)
        self.label_4.setFont(font)
        self.label_4.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_4.setObjectName("label_4")
        self.test_data = QtWidgets.QLineEdit(self.centralwidget)
        self.test_data.setGeometry(QtCore.QRect(900, 60, 71, 23))
        self.test_data.setObjectName("test_data")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(760, 60, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.test_size_btn = QtWidgets.QPushButton(self.centralwidget)
        self.test_size_btn.setGeometry(QtCore.QRect(900, 100, 71, 23))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.test_size_btn.setFont(font)
        self.test_size_btn.setObjectName("test_size_btn")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(420, 250, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(10, 250, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setStyleSheet("background-color: rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);\n"
"")
        self.label_6.setObjectName("label_6")
        self.randomstate = QtWidgets.QLineEdit(self.centralwidget)
        self.randomstate.setGeometry(QtCore.QRect(560, 320, 111, 23))
        self.randomstate.setObjectName("randomstate")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(430, 320, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        self.predict.setGeometry(QtCore.QRect(430, 560, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.predict.setFont(font)
        self.predict.setObjectName("predict")
        self.label_16 = QtWidgets.QLabel(self.centralwidget)
        self.label_16.setGeometry(QtCore.QRect(10, 520, 251, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.ExitButton = QtWidgets.QPushButton(self.centralwidget)
        self.ExitButton.setGeometry(QtCore.QRect(880, 600, 91, 31))
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
        self.dwnld_2 = QtWidgets.QPushButton(self.centralwidget)
        self.dwnld_2.setGeometry(QtCore.QRect(650, 200, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.dwnld_2.setFont(font)
        self.dwnld_2.setObjectName("dwnld_2")
        self.label_19 = QtWidgets.QLabel(self.centralwidget)
        self.label_19.setGeometry(QtCore.QRect(730, 470, 111, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_19.setFont(font)
        self.label_19.setObjectName("label_19")
        self.predict_val = QtWidgets.QLabel(self.centralwidget)
        self.predict_val.setGeometry(QtCore.QRect(770, 510, 211, 41))
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
        self.label_20 = QtWidgets.QLabel(self.centralwidget)
        self.label_20.setGeometry(QtCore.QRect(440, 60, 131, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(700, 250, 20, 241))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.list = QtWidgets.QLineEdit(self.centralwidget)
        self.list.setGeometry(QtCore.QRect(220, 510, 471, 31))
        self.list.setObjectName("list")
        self.split = QtWidgets.QLabel(self.centralwidget)
        self.split.setGeometry(QtCore.QRect(880, 130, 141, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.split.setFont(font)
        self.split.setStyleSheet("color: rgb(1, 78, 144);")
        self.split.setText("")
        self.split.setObjectName("split")
        self.target = QtWidgets.QLabel(self.centralwidget)
        self.target.setGeometry(QtCore.QRect(850, 470, 141, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.target.setFont(font)
        self.target.setObjectName("target")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 31))
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
        self.train.setText(_translate("MainWindow", "Train"))
        self.max_iter.setText(_translate("MainWindow", "100"))
        self.conf_mat.setText(_translate("MainWindow", "Confusion Matrix"))
        self.label_7.setText(_translate("MainWindow", "Max_Iter:"))
        self.output.setText(_translate("MainWindow", "Visualize Boundary"))
        self.solver.setItemText(0, _translate("MainWindow", "lbfgs"))
        self.solver.setItemText(1, _translate("MainWindow", "liblinear"))
        self.solver.setItemText(2, _translate("MainWindow", "sag"))
        self.solver.setItemText(3, _translate("MainWindow", "saga"))
        self.solver.setItemText(4, _translate("MainWindow", "netwon-cg"))
        self.label_22.setText(_translate("MainWindow", "Solver:"))
        self.multi_class.setItemText(0, _translate("MainWindow", "auto"))
        self.multi_class.setItemText(1, _translate("MainWindow", "ovr"))
        self.multi_class.setItemText(2, _translate("MainWindow", "multinomial"))
        self.label_23.setText(_translate("MainWindow", "multi_class:"))
        self.label_13.setText(_translate("MainWindow", "Mean Square Error:"))
        self.label_14.setText(_translate("MainWindow", "Root Mean Sq. Error:"))
        self.label_15.setText(_translate("MainWindow", "Mean Absolute Error:"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Logistic Regression Model</p></body></html>"))
        self.test_data.setText(_translate("MainWindow", "0.1"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p>Test Train Split</p></body></html>"))
        self.test_size_btn.setText(_translate("MainWindow", "Set"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Hyper Parameters</p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\">Classification Report:</p><p align=\"center\"><br/></p><p align=\"center\"><br/></p></body></html>"))
        self.randomstate.setText(_translate("MainWindow", "144"))
        self.label_9.setText(_translate("MainWindow", "Random State:"))
        self.predict.setText(_translate("MainWindow", "Predict"))
        self.label_16.setText(_translate("MainWindow", "Enter List for Prediction  :"))
        self.ExitButton.setText(_translate("MainWindow", "Exit"))
        self.dwnld_2.setText(_translate("MainWindow", "Download Model"))
        self.label_19.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:14pt; font-weight:600;\">Predicted </span></p></body></html>"))
        self.predict_val.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.label_20.setText(_translate("MainWindow", "Accuracy :"))
        self.list.setText(_translate("MainWindow", "0.1"))
        self.target.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600;\"><br/></span></p></body></html>"))
