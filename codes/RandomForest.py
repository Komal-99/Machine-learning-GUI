# importing Libraries
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
import common as common
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
class UI(QMainWindow):  # inherit from QMainWindow  
    def __init__(self,df_original,df,target,user_actions):  # constructor
        super(UI, self).__init__()
        uic.loadUi("ui_files/RandomForest.ui", self)    # load the UI
        self.user_act=user_actions  # user_actions is the object of class User_Actions
        global data ,steps  # data is the dataframe
        data=data_visualise.data_() # data is the dataframe
        steps=common.common_steps(df,target)    # steps is the object of class Common_steps
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()    # X is the dataframe    
        # n_classes is the number of classes  
        # target_value is the target value    
        # df is the dataframe    
        # column_list is the list of columns   of the dataframe

        # defining the functions and its labels
        self.columns= self.findChild(QListWidget,"columns")
        self.random=self.findChild(QLineEdit,"randomstate")
        self.estimators=self.findChild(QLineEdit,"estimators")
        self.criterion=self.findChild(QComboBox,"criterion")
        self.max_depth=self.findChild(QLineEdit,"max_depth")
        self.min_sample_split=self.findChild(QLineEdit,"min_sample_split")
        self.bootstrap=self.findChild(QComboBox,"bootstrap")
        self.train_btn=self.findChild(QPushButton,"train")
        self.split_done= self.findChild(QLabel,"split")
        self.target=self.findChild(QLabel,"target")
        
        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mse")
        self.rmse=self.findChild(QLabel,"rmse")
        self.accuracy=self.findChild(QLabel,"accuracy")
        self.exitbutton = self.findChild(QPushButton,"pushButton")

        self.exitbutton.clicked.connect(self.exit)

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
        self.train_btn.setStyleSheet(
                             "QPushButton::pressed"
                             "{"
                             "background-color : green;"
                             "}"
                             )  # setting the style of the button
        
        self.dwnld.clicked.connect(self.download_model) # download the model
        self.setvalue() # set the values of the columns
        self.show()     # show the window

    def exit(self): # exit function
        sys.exit()
    def set_predict(self):  # function to predict the value
        self.a = self.list.text()   # a is the value of the list
        self.ls = self.a.split(",") # ls is the list of the values
        self.target.setText(str(self.target_value)) # set the target value
        self.ls_updated = [float(x) for x in self.ls]   # converting the list to float
        self.ls_array =  np.array(self.ls_updated)  # converting the list to array

        self.pred  =self.rf.predict([self.ls_array])    # pred is the predicted value
        self.predict_val.setText(str(self.pred))    # set the predicted value


    def setvalue(self): # function to set the values of the columns
        self.columns.addItems(self.column_list) # add the columns to the list


    def download_model(self):   # function to download the model

        try:

            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','',"pickle(*.pkl)")
            pkl_filename = name[0]      # pkl_filename is the name of the file
            with open(pkl_filename, 'wb') as file:  # open the file
                model= (self.rf, accuracy_score(self.pre,self.y_test))
                pickle.dump(model, file) 


            self.user_act.save_file(pkl_filename)  

        except:

            self.w = error_window()
            self.w.errortype.setText("Failed to save the file")
            self.w.show()
    
    def test_split(self):   # function to split the data

        try:

            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
            print(self.y_train.shape)   # print the shape of the train and test data
            print(self.y_test.shape)
            self.split_done.setText(str("Split Done"))  # set the text of the label

        except:

            self.w = error_window()
            self.w.errortype.setText("Size Not set")
            self.w.show()
        
    def training(self):  # function to train the model

        try:

            self.rf = RFC(n_estimators=int(self.estimators.text()),criterion=self.criterion.currentText(),max_depth=None,min_samples_split=int(self.min_sample_split.text()),bootstrap=self.bootstrap.currentText()=='True',random_state=1) # create the model of Randomm forest

            self.rf.fit(self.x_train,self.y_train)  # fit the model
            
            self.pre=self.rf.predict(self.x_test)   # predict the value
            self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))    # set the mae
            self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))   # set the mse
            self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))   # set the rmse
            self.accuracy.setText(str(accuracy_score(self.pre,self.y_test)))    # set the accuracy
            text=steps.classification_(self.y_test,self.pre)    # get the classification report
            self.report.setPlainText(text)  

        except:

            self.w =error_window()
            self.w.errortype.setText("First Split your dataset!")
            self.w.show()

    def conf_matrix(self):

        try:

            data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }      # create the dataframe
            df = pd.DataFrame(data, columns=['y_Actual','y_Predicted']) 
            confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])  # create the confusion matrix
            plt.figure()    
            sns.heatmap(confusion_matrix, annot=True)   # plot the confusion matrix
            plt.show()  # show the plot

        except:

            self.w =error_window()
            self.w.errortype.setText("Train Your Model First!")
            self.w.show()

