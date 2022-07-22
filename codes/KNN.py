
# Importing libraries
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser, QDialog
import sys ,pickle
import data_visualise
import table_display
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
import common

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
class UI(QDialog): # QDialog is the base class  of all user interface objects in PyQt5. 
    def __init__(self,df_original,df,target,user_actions):  # Constructor
        super(UI, self).__init__()
        uic.loadUi("ui_files/KNN.ui", self) # Load the UI
        self.user_act=user_actions  # User Actions
        global data ,steps  # Global variables
        data=data_visualise.data_() # Data Visualise
        steps=common.common_steps(df,target)    # Common Steps
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()    # Return Data
        #defining the variables and their respective functions
        self.columns= self.findChild(QListWidget,"columns")     # Find the child  of the object 
        self.solver=self.findChild(QComboBox,"solver")        
        self.random=self.findChild(QLineEdit,"randomstate")     
        self.max_iter=self.findChild(QLineEdit,"max_iter")  
        self.multi_class=self.findChild(QComboBox,"multi_class")
        self.train_btn=self.findChild(QPushButton,"train")
        
        self.mae=self.findChild(QLabel,"mae_2")
        self.mse=self.findChild(QLabel,"mae_4")
        self.rmse=self.findChild(QLabel,"rmse")
        self.accuracy=self.findChild(QLabel,"accuracy_score")
        self.target=self.findChild(QLabel,"target")
        self.split_done= self.findChild(QLabel,"split")
        
        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        # self.roc_btn.clicked.connect(self.roc_plot)
        self.conf_mat_btn.clicked.connect(self.conf_matrix)
        self.test_size_btn.clicked.connect(self.test_split)
        self.dwnld_2.clicked.connect(self.download_model)
        self.exitbutton = self.findChild(QPushButton,"pushButton")

        self.exitbutton.clicked.connect(self.exit)
        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_predict)
        self.train_btn.setStyleSheet(   
                             "QPushButton::pressed"
                             "{"
                             "background-color : green;"
                             "}"
                             )  # Set the style sheet for the train  button
        
        self.setvalue() # Set the values of the columns
        self.show() # Show the window

    def setvalue(self): # Set the values of the columns
        self.columns.addItems(self.column_list)
    def exit(self): # Exit the window
        sys.exit()  

   
    def set_predict(self):  # Set the predict value function
        self.a = self.list.text()   # Get the value from the text box
        self.ls = self.a.split(",") # Split the value
       
        self.ls_updated = [float(x) for x in self.ls]   # Convert the string to float
        self.ls_array =  np.array(self.ls_updated)  # Convert the list to array
        self.target.setText(str(self.target_value)) # Set the target value
        self.pred  =self.knn.predict([self.ls_array])   # Predict the value
        self.predict_val.setText(np.str_(self.pred))    # Set the predicted value


    
    def test_split(self):   # Test Split function

        try:

            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
            print(self.y_train.shape)   # Print the shape of the train and test data
            print(self.y_test.shape)
            self.split_done.setText(str("Split Done"))  # Set the text of the label

        except:

            self.w =error_window()
            self.w.errortype.setText("Size Not set")
            self.w.show()
   

    def download_model(self):   # Download the model function

        try:

            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")  # Get the file name
            pkl_filename = name[0]  # Get the file name
            with open(pkl_filename, 'wb') as file:  # Open the file
                pickle.dump(self.knn, file)     # Write the model to the file

            self.user_act.save_file(pkl_filename)   # Save the file  

        except:

            self.w = error_window()
            self.w.errortype.setText("Failed to save the file")
            self.w.show()

    def training(self):  # Training function

        try:

            self.knn = KNC(n_neighbors=int(self.neighbours.text()),weights=self.weights.currentText(),algorithm=self.algorithm.currentText())   # Create the model knn
            self.knn.fit(self.x_train,self.y_train)  # Fit the model
            self.pre=self.knn.predict(self.x_test)  # Predict the value
            self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))    # Set the mean absolute error
            self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))   # Set the mean squared error
            self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))   # Set the root mean squared error
            self.accuracy.setText(str(accuracy_score(self.pre,self.y_test)))    # Set the accuracy score
            text=steps.classification_(self.y_test,self.pre)    # Get the classification report
            self.report.setPlainText(text)  # Set the text of the report

        except:

            self.w =error_window()
            self.w.errortype.setText("First Split your dataset!")
            self.w.show()

    def conf_matrix(self):  # Confusion Matrix function

        try:

            data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }      # Create the dataframe
            df = pd.DataFrame(data, columns=['y_Actual','y_Predicted']) # Create the dataframe
            confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])  # Create the confusion matrix
            plt.figure()    
            sns.heatmap(confusion_matrix, annot=True)
            plt.show()  # Show the confusion matrix

        except:

            self.w =error_window()
            self.w.errortype.setText("Train Your Model First!")
            self.w.show()

   