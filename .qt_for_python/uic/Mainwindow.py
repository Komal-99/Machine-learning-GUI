# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\Users\aaa\OneDrive - The Northcap university\Desktop\UTS\UTS-Project\ui_files\Mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1408, 920)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 60, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setWordWrap(False)
        self.label.setObjectName("label")
        self.Browse = QtWidgets.QPushButton(self.centralwidget)
        self.Browse.setGeometry(QtCore.QRect(160, 60, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.Browse.setFont(font)
        self.Browse.setObjectName("Browse")
        self.Submit = QtWidgets.QPushButton(self.centralwidget)
        self.Submit.setGeometry(QtCore.QRect(310, 60, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.Submit.setFont(font)
        self.Submit.setObjectName("Submit")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(670, 70, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(20, 350, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.dropcolumn = QtWidgets.QComboBox(self.centralwidget)
        self.dropcolumn.setGeometry(QtCore.QRect(150, 350, 191, 31))
        self.dropcolumn.setObjectName("dropcolumn")
        self.Drop = QtWidgets.QPushButton(self.centralwidget)
        self.Drop.setGeometry(QtCore.QRect(360, 350, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.Drop.setFont(font)
        self.Drop.setObjectName("Drop")
        self.heatmap = QtWidgets.QPushButton(self.centralwidget)
        self.heatmap.setGeometry(QtCore.QRect(100, 480, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.heatmap.setFont(font)
        self.heatmap.setObjectName("heatmap")
        self.column_list = QtWidgets.QListWidget(self.centralwidget)
        self.column_list.setGeometry(QtCore.QRect(20, 100, 341, 221))
        self.column_list.setObjectName("column_list")
        self.tableView = QtWidgets.QTableView(self.centralwidget)
        self.tableView.setGeometry(QtCore.QRect(670, 100, 691, 221))
        self.tableView.setObjectName("tableView")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(890, 350, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_5.setFont(font)
        self.label_5.setScaledContents(False)
        self.label_5.setWordWrap(False)
        self.label_5.setObjectName("label_5")
        self.convert_btn = QtWidgets.QPushButton(self.centralwidget)
        self.convert_btn.setGeometry(QtCore.QRect(1240, 350, 101, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.convert_btn.setFont(font)
        self.convert_btn.setObjectName("convert_btn")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 490, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(20, 530, 161, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(230, 520, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(530, 460, 191, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.scatterplot = QtWidgets.QPushButton(self.centralwidget)
        self.scatterplot.setGeometry(QtCore.QRect(60, 760, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.scatterplot.setFont(font)
        self.scatterplot.setObjectName("scatterplot")
        self.histogram = QtWidgets.QPushButton(self.centralwidget)
        self.histogram.setGeometry(QtCore.QRect(250, 760, 81, 23))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.histogram.setFont(font)
        self.histogram.setObjectName("histogram")
        self.shape = QtWidgets.QLabel(self.centralwidget)
        self.shape.setGeometry(QtCore.QRect(670, 20, 351, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.shape.setFont(font)
        self.shape.setObjectName("shape")
        self.scatter_x = QtWidgets.QComboBox(self.centralwidget)
        self.scatter_x.setGeometry(QtCore.QRect(100, 560, 81, 23))
        self.scatter_x.setObjectName("scatter_x")
        self.scatter_y = QtWidgets.QComboBox(self.centralwidget)
        self.scatter_y.setGeometry(QtCore.QRect(100, 590, 81, 23))
        self.scatter_y.setObjectName("scatter_y")
        self.scatter_mark = QtWidgets.QComboBox(self.centralwidget)
        self.scatter_mark.setGeometry(QtCore.QRect(100, 650, 81, 23))
        self.scatter_mark.setObjectName("scatter_mark")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.scatter_mark.addItem("")
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setGeometry(QtCore.QRect(20, 560, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_12.setFont(font)
        self.label_12.setWordWrap(False)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.centralwidget)
        self.label_13.setGeometry(QtCore.QRect(20, 590, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_13.setFont(font)
        self.label_13.setWordWrap(False)
        self.label_13.setObjectName("label_13")
        self.label_14 = QtWidgets.QLabel(self.centralwidget)
        self.label_14.setGeometry(QtCore.QRect(20, 620, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_14.setFont(font)
        self.label_14.setWordWrap(False)
        self.label_14.setObjectName("label_14")
        self.label_15 = QtWidgets.QLabel(self.centralwidget)
        self.label_15.setGeometry(QtCore.QRect(20, 650, 71, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_15.setFont(font)
        self.label_15.setWordWrap(False)
        self.label_15.setObjectName("label_15")
        self.scatter_c = QtWidgets.QComboBox(self.centralwidget)
        self.scatter_c.setGeometry(QtCore.QRect(100, 620, 81, 23))
        self.scatter_c.setObjectName("scatter_c")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(20, 400, 1341, 20))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(460, 410, 16, 431))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(1390, 520, 16, 301))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.hist_column = QtWidgets.QComboBox(self.centralwidget)
        self.hist_column.setGeometry(QtCore.QRect(230, 560, 79, 23))
        self.hist_column.setObjectName("hist_column")
        self.hist_add_btn = QtWidgets.QPushButton(self.centralwidget)
        self.hist_add_btn.setGeometry(QtCore.QRect(320, 560, 61, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.hist_add_btn.setFont(font)
        self.hist_add_btn.setObjectName("hist_add_btn")
        self.hist_column_add = QtWidgets.QComboBox(self.centralwidget)
        self.hist_column_add.setGeometry(QtCore.QRect(230, 600, 79, 23))
        self.hist_column_add.setObjectName("hist_column_add")
        self.hist_remove_btn = QtWidgets.QPushButton(self.centralwidget)
        self.hist_remove_btn.setGeometry(QtCore.QRect(320, 600, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.hist_remove_btn.setFont(font)
        self.hist_remove_btn.setObjectName("hist_remove_btn")
        self.label_24 = QtWidgets.QLabel(self.centralwidget)
        self.label_24.setGeometry(QtCore.QRect(230, 640, 91, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_24.setFont(font)
        self.label_24.setWordWrap(False)
        self.label_24.setObjectName("label_24")
        self.hist_bin = QtWidgets.QLineEdit(self.centralwidget)
        self.hist_bin.setGeometry(QtCore.QRect(320, 640, 81, 23))
        self.hist_bin.setObjectName("hist_bin")
        self.hist_x = QtWidgets.QLineEdit(self.centralwidget)
        self.hist_x.setGeometry(QtCore.QRect(320, 670, 81, 23))
        self.hist_x.setObjectName("hist_x")
        self.hist_y = QtWidgets.QLineEdit(self.centralwidget)
        self.hist_y.setGeometry(QtCore.QRect(320, 700, 81, 23))
        self.hist_y.setObjectName("hist_y")
        self.label_25 = QtWidgets.QLabel(self.centralwidget)
        self.label_25.setGeometry(QtCore.QRect(230, 670, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_25.setFont(font)
        self.label_25.setWordWrap(False)
        self.label_25.setObjectName("label_25")
        self.label_26 = QtWidgets.QLabel(self.centralwidget)
        self.label_26.setGeometry(QtCore.QRect(230, 700, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_26.setFont(font)
        self.label_26.setWordWrap(False)
        self.label_26.setObjectName("label_26")
        self.train = QtWidgets.QPushButton(self.centralwidget)
        self.train.setGeometry(QtCore.QRect(550, 550, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.train.setFont(font)
        self.train.setObjectName("train")
        self.model_select = QtWidgets.QComboBox(self.centralwidget)
        self.model_select.setGeometry(QtCore.QRect(530, 510, 181, 23))
        self.model_select.setObjectName("model_select")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.model_select.addItem("")
        self.label_27 = QtWidgets.QLabel(self.centralwidget)
        self.label_27.setGeometry(QtCore.QRect(610, 610, 81, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_27.setFont(font)
        self.label_27.setWordWrap(False)
        self.label_27.setObjectName("label_27")
        self.label_28 = QtWidgets.QLabel(self.centralwidget)
        self.label_28.setGeometry(QtCore.QRect(530, 640, 211, 21))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.pre_trained = QtWidgets.QPushButton(self.centralwidget)
        self.pre_trained.setGeometry(QtCore.QRect(540, 670, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.pre_trained.setFont(font)
        self.pre_trained.setObjectName("pre_trained")
        self.go_pre_trained = QtWidgets.QPushButton(self.centralwidget)
        self.go_pre_trained.setGeometry(QtCore.QRect(640, 670, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.go_pre_trained.setFont(font)
        self.go_pre_trained.setObjectName("go_pre_trained")
        self.label_29 = QtWidgets.QLabel(self.centralwidget)
        self.label_29.setGeometry(QtCore.QRect(460, 90, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_29.setFont(font)
        self.label_29.setWordWrap(False)
        self.label_29.setObjectName("label_29")
        self.target_col = QtWidgets.QLabel(self.centralwidget)
        self.target_col.setGeometry(QtCore.QRect(460, 160, 201, 61))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.target_col.setFont(font)
        self.target_col.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.target_col.setStatusTip("")
        self.target_col.setAutoFillBackground(False)
        self.target_col.setStyleSheet("color:rgb(1, 78, 144)")
        self.target_col.setObjectName("target_col")
        self.cat_column = QtWidgets.QComboBox(self.centralwidget)
        self.cat_column.setGeometry(QtCore.QRect(1090, 350, 141, 31))
        self.cat_column.setObjectName("cat_column")
        self.scale_btn = QtWidgets.QPushButton(self.centralwidget)
        self.scale_btn.setGeometry(QtCore.QRect(1230, 20, 131, 31))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.scale_btn.setFont(font)
        self.scale_btn.setObjectName("scale_btn")
        self.scaler = QtWidgets.QComboBox(self.centralwidget)
        self.scaler.setGeometry(QtCore.QRect(1050, 20, 161, 31))
        self.scaler.setObjectName("scaler")
        self.scaler.addItem("")
        self.scaler.addItem("")
        self.scaler.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(0, 420, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_2.setWordWrap(False)
        self.label_2.setObjectName("label_2")
        self.label_30 = QtWidgets.QLabel(self.centralwidget)
        self.label_30.setGeometry(QtCore.QRect(0, 10, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_30.setFont(font)
        self.label_30.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);\n"
"\n"
"")
        self.label_30.setWordWrap(False)
        self.label_30.setObjectName("label_30")
        self.label_31 = QtWidgets.QLabel(self.centralwidget)
        self.label_31.setGeometry(QtCore.QRect(470, 420, 251, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu Condensed")
        font.setPointSize(11)
        self.label_31.setFont(font)
        self.label_31.setStyleSheet("background-color:rgb(1, 78, 144);\n"
"color: rgb(255, 255, 255);")
        self.label_31.setWordWrap(False)
        self.label_31.setObjectName("label_31")
        self.plotwidget = QtWidgets.QWidget(self.centralwidget)
        self.plotwidget.setGeometry(QtCore.QRect(750, 460, 351, 371))
        self.plotwidget.setObjectName("plotwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1408, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.actionData_Visualisation = QtWidgets.QAction(MainWindow)
        self.actionData_Visualisation.setObjectName("actionData_Visualisation")
        self.actionTrain_Data = QtWidgets.QAction(MainWindow)
        self.actionTrain_Data.setObjectName("actionTrain_Data")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Select Data File:"))
        self.Browse.setText(_translate("MainWindow", "Browse"))
        self.Submit.setText(_translate("MainWindow", "Set Target"))
        self.label_3.setText(_translate("MainWindow", "Data Details:"))
        self.label_4.setText(_translate("MainWindow", "Drop Columns:"))
        self.Drop.setText(_translate("MainWindow", "Drop"))
        self.heatmap.setText(_translate("MainWindow", "Show"))
        self.label_5.setText(_translate("MainWindow", "Encode Categorical Data:"))
        self.convert_btn.setText(_translate("MainWindow", "Convert"))
        self.label_6.setText(_translate("MainWindow", "Heatmap"))
        self.label_7.setText(_translate("MainWindow", "Scatter Plot"))
        self.label_9.setText(_translate("MainWindow", "Histogram :  Select Columns"))
        self.label_10.setText(_translate("MainWindow", "Select Training Model"))
        self.scatterplot.setText(_translate("MainWindow", "Show"))
        self.histogram.setText(_translate("MainWindow", "Show"))
        self.shape.setText(_translate("MainWindow", "Shape: "))
        self.scatter_mark.setItemText(0, _translate("MainWindow", "."))
        self.scatter_mark.setItemText(1, _translate("MainWindow", ","))
        self.scatter_mark.setItemText(2, _translate("MainWindow", "o"))
        self.scatter_mark.setItemText(3, _translate("MainWindow", "v"))
        self.scatter_mark.setItemText(4, _translate("MainWindow", "^"))
        self.scatter_mark.setItemText(5, _translate("MainWindow", "<"))
        self.scatter_mark.setItemText(6, _translate("MainWindow", ">"))
        self.scatter_mark.setItemText(7, _translate("MainWindow", "-"))
        self.scatter_mark.setItemText(8, _translate("MainWindow", "1"))
        self.scatter_mark.setItemText(9, _translate("MainWindow", "2"))
        self.scatter_mark.setItemText(10, _translate("MainWindow", "3"))
        self.scatter_mark.setItemText(11, _translate("MainWindow", "4"))
        self.scatter_mark.setItemText(12, _translate("MainWindow", "8"))
        self.label_12.setText(_translate("MainWindow", " X-Axis:"))
        self.label_13.setText(_translate("MainWindow", " Y-Axis:"))
        self.label_14.setText(_translate("MainWindow", " Color:"))
        self.label_15.setText(_translate("MainWindow", "marker  :"))
        self.hist_add_btn.setText(_translate("MainWindow", "Add"))
        self.hist_remove_btn.setText(_translate("MainWindow", "Remove"))
        self.label_24.setText(_translate("MainWindow", "Bins :"))
        self.label_25.setText(_translate("MainWindow", "Figsize_x :"))
        self.label_26.setText(_translate("MainWindow", "Figsize_y :"))
        self.train.setText(_translate("MainWindow", "Train"))
        self.model_select.setItemText(0, _translate("MainWindow", "Linear Regression"))
        self.model_select.setItemText(1, _translate("MainWindow", "SVM"))
        self.model_select.setItemText(2, _translate("MainWindow", "SVR"))
        self.model_select.setItemText(3, _translate("MainWindow", "Logistic Regression"))
        self.model_select.setItemText(4, _translate("MainWindow", "Random Forest"))
        self.model_select.setItemText(5, _translate("MainWindow", "K-Nearest Neighbour"))
        self.model_select.setItemText(6, _translate("MainWindow", "Gaussian NB"))
        self.model_select.setItemText(7, _translate("MainWindow", "Multi Layer Perceptron"))
        self.label_27.setText(_translate("MainWindow", "Or"))
        self.label_28.setText(_translate("MainWindow", "Use a Pre-Trained Model :"))
        self.pre_trained.setText(_translate("MainWindow", "Browse"))
        self.go_pre_trained.setText(_translate("MainWindow", "Go"))
        self.label_29.setText(_translate("MainWindow", "Target Set :"))
        self.target_col.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><br/></p><p align=\"center\"><br/></p></body></html>"))
        self.scale_btn.setText(_translate("MainWindow", "Scale Values"))
        self.scaler.setItemText(0, _translate("MainWindow", "StandardScale"))
        self.scaler.setItemText(1, _translate("MainWindow", "MinMaxScale"))
        self.scaler.setItemText(2, _translate("MainWindow", "PowerScale"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600;\">Visualizations</span></p></body></html>"))
        self.label_30.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600;\">Data Preparation</span></p></body></html>"))
        self.label_31.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:16pt; font-weight:600;\">Train/Test</span></p></body></html>"))
        self.actionData_Visualisation.setText(_translate("MainWindow", "Data Visualisation"))
        self.actionTrain_Data.setText(_translate("MainWindow", "Train Data"))
