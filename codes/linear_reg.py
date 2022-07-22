from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys,pickle
from PyQt5.QtCore import QCoreApplication
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from pyparsing import null_debug_action
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import data_visualise   
import pandas as pd
import common
from sklearn.preprocessing import OrdinalEncoder
import uicode

# MainWindow of the Linear Regression Model
class UI(QMainWindow):
    def __init__(self,df_original,df,target,user_actions):
        super(UI, self).__init__()
        uic.loadUi('ui_files\LinearRegression.ui', self)
        self.user_act=user_actions
        global data 
        #Calling the data_ class from data_visualise.py
        data=data_visualise.data_()
        #Calling the common class from common_steps.py
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()



        #Identifying the required Buttons from QTDesigner
        self.columns= self.findChild(QListWidget,"columns")
        self.split_done= self.findChild(QLabel,"split")
        self.list=self.findChild(QLineEdit,"list")
        self.test_data=self.findChild(QLineEdit,"test_data")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.intercept=self.findChild(QLabel,"intercept")
        self.error = self.findChild(QLabel,"error")
        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mse")
        self.rmse=self.findChild(QLabel,"rmse")
        self.target=self.findChild(QLabel,"target")
        self.fit_inter =self.findChild(QComboBox,"fit_inter")
        self.normalize=self.findChild(QComboBox,"normalize")
        self.predict_btn=self.findChild(QPushButton,"train_2")
        self.exitbutton = self.findChild(QPushButton,"ExitButton")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.bar_plot_btn=self.findChild(QPushButton,"bar_plot")
        self.train_btn=self.findChild(QPushButton,"train")
        self.weights=self.findChild(QTextBrowser,"weights")


        #Connecting the buttons with the coresponding functions
        self.exitbutton.clicked.connect(self.exit)
        self.test_size_btn.clicked.connect(self.test_split)
        self.train_btn.clicked.connect(self.training)
        self.bar_plot_btn.clicked.connect(self.barplot)
        self.dwnld.clicked.connect(self.download_model)
        self.predict_btn.clicked.connect(self.set_valpred)

        self.train_btn.setStyleSheet(
                             "QPushButton::pressed"
                             "{"
                             "background-color : green;"
                             "}"
                             )
        self.show()
        self.setvalue()

    # Setting the the columns value
    def exit(self):
        sys.exit()
    def setvalue(self):
        self.columns.addItems(self.column_list)
    
    # Prediction for a Specific Value
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
                self.pred  =self.reg.predict([self.ls_array])
                self.predict_val.setText(str(self.pred))
        except:
            self.w = uicode.error_window()
            self.w.errortype.setText("Failed to Predict")
            self.w.show()

    #Downloading a Trained Model
    def download_model(self):
        try:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','',"pickle(*.pkl)")
            pkl_filename = name[0]
            with open(pkl_filename, 'wb') as file:
                pickle.dump(self.reg, file)  
            self.user_act.save_file(pkl_filename)  
        except:
            self.w = uicode.error_window()
            self.w.errortype.setText("Failed to save the file")
            self.w.show()

    #Splitting the Model
    def test_split(self):
        try:
            
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=0)
            print(self.y_train.shape)
            print(self.y_test.shape)
            self.split_done.setText(str("Split Done"))
        
        except:

            self.w =uicode.error_window()
            self.w.errortype.setText("Size Not set")
            self.w.show()
     
    #Training the Model
    def training(self):

        try:
            self.reg=LinearRegression().fit(self.x_train,self.y_train)
            coef=' '.join(map(str, self.reg.coef_)) 
            self.intercept.setText(str(self.reg.intercept_))
            self.weights.setText(coef)

            #Various Parameters for the trained model
            pre=self.reg.predict(self.x_test)
            self.mae.setText(str(metrics.mean_absolute_error(self.y_test,pre)))
            self.mse.setText(str(metrics.mean_squared_error(self.y_test,pre)))
            self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,pre))))
            self.accuracy.setText(str(self.reg.score(self.x_test,self.y_test)))

        except:

            self.w =uicode.error_window()
            self.w.errortype.setText("First Split your dataset!")
            self.w.show() 
            
    #plotting the Bar Graph
    def barplot(self):

        try:
            y_pred = self.reg.predict(self.x_test)
            df = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})
            df1=df.head(20)
            
            df1.plot(kind='bar')
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            plt.show()

        except:

            self.w =uicode.error_window()
            self.w.errortype.setText("Train Your Model First!")
            self.w.show()