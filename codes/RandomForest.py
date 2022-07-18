
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys,pickle
import data_visualise
from itertools import cycle
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.metrics import accuracy_score , confusion_matrix , roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,auc
import common

class UI(QMainWindow):
    def __init__(self,df,target,user_actions):
        super(UI, self).__init__()
        uic.loadUi("ui_files/RandomForest.ui", self)
        self.user_act=user_actions
        global data ,steps
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        self.columns= self.findChild(QListWidget,"columns")
        self.random=self.findChild(QLineEdit,"randomstate")
        self.estimators=self.findChild(QLineEdit,"estimators")
        self.criterion=self.findChild(QComboBox,"criterion")
        self.max_depth=self.findChild(QLineEdit,"max_depth")
        self.min_sample_split=self.findChild(QLineEdit,"min_sample_split")
        self.bootstrap=self.findChild(QComboBox,"bootstrap")
        self.train_btn=self.findChild(QPushButton,"train")
        
        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mse")
        self.rmse=self.findChild(QLabel,"rmse")
        self.accuracy=self.findChild(QLabel,"accuracy")

        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_predict)
        self.conf_mat_btn.clicked.connect(self.conf_matrix)
        self.test_size_btn.clicked.connect(self.test_split)
        
        self.dwnld.clicked.connect(self.download_model)
        self.setvalue()
        self.show()

       
    def set_predict(self):
        self.a = self.list.text()
        self.ls = self.a.split(",")
       
        self.ls_updated = [float(x) for x in self.ls]
        self.ls_array =  np.array(self.ls_updated)

        self.pred  =self.lr.predict([self.ls_array])
        self.predict_val.setText(str(self.pred))


    def setvalue(self):
        self.columns.addItems(self.column_list)
    

    def download_model(self):

        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
        pkl_filename = name[0]
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.lr, file)  
        
        self.user_act.save_file(pkl_filename)  
    
    def test_split(self):

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
        print(self.y_train.shape)
        print(self.y_test.shape)
        
    def training(self):

        self.lr = RFC(n_estimators=int(self.estimators.text()),criterion=self.criterion.currentText(),max_depth=None,min_samples_split=int(self.min_sample_split.text()),bootstrap=self.bootstrap.currentText()=='True',random_state=1)
        self.lr.fit(self.x_train,self.y_train)
        
        self.pre=self.lr.predict(self.x_test)
        self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))
        self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))
        self.accuracy.setText(str(accuracy_score(self.pre,self.y_test)))
        text=steps.classification_(self.y_test,self.pre)
        self.report.setPlainText(text)

    def conf_matrix(self):

        data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

