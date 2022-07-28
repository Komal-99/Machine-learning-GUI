import os
import sys
from turtle import goto
from PyQt5.QtWidgets import *
from os import system
import re,pickle
import data_visualise,common as common

from PyQt5 import uic, QtWidgets ,QtCore, QtGui

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class error_window(QMainWindow): #error window class
    def __init__(self): #constructor
        super(error_window, self).__init__()        
        uic.loadUi("ui_files/error.ui", self)
        self.ExitError = self.findChild(QPushButton, "ExitButtonError") #exit button
        self.ExitError.clicked.connect(self.exit)
        self.back = self.findChild(QPushButton,"Back")  
        self.errortype = self.findChild(QLabel, 'Error_type')     
        self.back.clicked.connect(self.Backbut) #back button
        self.show() #show the window
#  Home Screen class to start our project
    def exit(self): #exit button
        sys.exit()  # exit the application
    def Backbut(self):  #back button
        self.back.clicked.connect(UI().target)
        self.close()    # close the window



class UI(QMainWindow):
    def __init__(self,df,df_original,target_value,pickle_model,path):
        super(UI, self).__init__()
        uic.loadUi("ui_files/pre_trained.ui", self)
        self.path=path
        self.df_original = df_original
        global data 
        self.df=df
        self.target_value = target_value
        print(self.target_value)

        data=data_visualise.data_()
        steps=common.common_steps(self.df,self.target_value)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.test_size= self.findChild(QLabel,"test_size") 
        self.Failure_name = self.findChild(QLabel,"Failure_Name")
        self.model=pickle_model      #load the model
        self.conf_mat.clicked.connect(self.conf_matrix)
        self.test.clicked.connect(self.test_model)
        self.exitbutton = self.findChild(QPushButton,"ExitButton")
        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_valpred)
        self.target=self.findChild(QLabel,"target_2")
        self.exitbutton.clicked.connect(self.exit)
        self.setvalue()
        self.show()
    def exit(self):
        sys.exit()
    def setvalue(self):
        self.target.setText(str(self.target_value))
        self.columns.addItems(self.column_list)
        self.data_shape.setText(str(self.df.shape))
      
        st=str(self.path)
        s=st.split(".")[0]
        f= s+".txt"
        original=sys.stdout
        sys.stdout = open(f, 'w')
        print(self.model)
        sys.stdout=original
        text=open(f).read()
        self.model_details.setPlainText(text)
        # os.remove(f)

        self.user_actions.setPlainText(text)
    
        


    def conf_matrix(self):
        try:
            data = {'y_Actual':self.X[self.target_value],'y_Predicted':self.pre }
            df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
            confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
            plt.figure()
            sns.heatmap(confusion_matrix, annot=True)
            plt.show()
        except:
            self.w =error_window()
            self.error.setText("No data to plot!")
            self.w.show()	

    def test_model(self):
        try:

            try:
                self.pre=self.model.predict(self.df)
                self.mae.setText(str(metrics.mean_absolute_error(self.X[self.target_value],self.pre)))
                self.mse.setText(str(metrics.mean_squared_error(self.X[self.target_value],self.pre)))
                self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.X[self.target_value],self.pre))))
                x=self.X[self.target_value]
                self.accuracy.setText(str(metrics.r2_score(x,self.pre)))
            except:
                self.pre=self.model.predict(self.df)
                self.mae.setText(str(metrics.mean_absolute_error(self.X[self.target_value],self.pre)))
                self.mse.setText(str(metrics.mean_squared_error(self.X[self.target_value],self.pre)))
                self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.X[self.target_value],self.pre))))
                self.accuracy.setText(str(metrics.accuracy_score(self.X[self.target_value],self.pre)))
        except:
            self.w =error_window()
            self.error.setText("Error while printing data values")
            self.w.show()	

    def set_valpred(self):
        try:
            pred = str(self.list.text())
            if len(pred) == 0:
                self.error.setText("Enter Values to Predict!")
            else:
                self.a = self.list.text() 
                self.ls = self.a.split(",")
                self.target.setText(str(self.target_value))
        
                self.ls_updated = [float(x) for x in self.ls]
                self.ls_array =  np.array(self.ls_updated)
                self.pred  =self.model.predict([self.ls_array])
                self.predict_val.setText(str(self.pred))
        except:
            self.w =error_window()
            self.error.setText("Can't predict these inputs")
            self.w.show()	
