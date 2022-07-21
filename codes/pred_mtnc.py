#  importing libraries
from base64 import decode
from re import A
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys ,pickle
from PyQt5.QtCore import QCoreApplication
from uicode import *
import data_visualise
import table_display
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import common
import time
import plotly.express as px
import plotly.io as pio



class UI(QMainWindow): # inheriting QMainWindow class
    def __init__(self,df_original,df,target,user_actions):  # constructor
        super(UI, self).__init__()
        uic.loadUi("ui_files/PredictiveMaintenance.ui", self)   # loading the UI file
        self.df_original = df_original
        self.user_act=user_actions  # storing the user actions object
        global data ,steps  # global variables
        data=data_visualise.data_() # creating an object of data class
        steps=common.common_steps(df,target)    # creating an object of common steps class
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()

        # Defining buttons and there funtions
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        self.solver=self.findChild(QComboBox,"solver")        
   
        self.max_iter=self.findChild(QLineEdit,"max_iter")
        self.random_state=self.findChild(QLineEdit,"randomstate")
        self.fit_inter=self.findChild(QComboBox,"fit_inter")  
        self.multi_class=self.findChild(QComboBox,"multi_class")
        self.train_btn=self.findChild(QPushButton,"train")
        self.exitbutton = self.findChild(QPushButton,"pushButton")

        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mse")
        self.rmse=self.findChild(QLabel,"rmse")
        self.acc_sc=self.findChild(QLabel,"accuracy")
        self.download=self.findChild(QPushButton,"dwnld")
        self.X_combo=self.findChild(QComboBox,"X_combo")
        self.Y_combo=self.findChild(QComboBox,"Y_combo")
        self.Z_combo=self.findChild(QComboBox,"Z_combo")
        self.color_combo=self.findChild(QComboBox,"color_combo")

        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        self.plot3d_btn= self.findChild(QPushButton,"visualize")
        self.split_done= self.findChild(QLabel,"split")
        self.target=self.findChild(QLabel,"target")
        
        self.exitbutton.clicked.connect(self.exit)
        self.train_btn.clicked.connect(self.training) 
        self.conf_mat_btn.clicked.connect(self.conf_matrix) 
        self.test_size_btn.clicked.connect(self.test_split) 
        self.dwnld.clicked.connect(self.download_model)
        self.visualize.clicked.connect(self.plt3d)  

        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_predict)
        self.train_btn.setStyleSheet(
                             "QPushButton::pressed"
                             "{"
                             "background-color : green;"
                             "}"
                             )  # setting the style of the button
        self.setvalue() # calling the function to set the values in the combo boxes
        self.show() # showing the window
    def exit(self): # function to exit the program
        sys.exit()

    def setvalue(self): # function to set the values in the combo boxes
      
        self.columns.addItems(self.column_list) # adding the columns to the list
        # self.column_list= data.get_column_list(self.df)   # getting the column list
        self.X_combo.clear()    # clearing the combo box
        self.X_combo.addItems(self.column_list) # adding the columns to the combo box
        self.Y_combo.clear()    # clearing the combo box
        self.Y_combo.addItems(self.column_list)
        self.Z_combo.clear()    
        self.Z_combo.addItems(self.column_list) 
        self.color_combo.clear()
        self.color_combo.addItems(self.column_list)

    def test_split(self):   # function to split the data into test and train
        try:
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=0)
            print(self.y_train.shape)   # printing the shape of the train and test data
            print(self.y_test.shape)    
            self.split_done.setText(str("Split Done"))  # setting the text of the label
        except:
                self.w =error_window()
                self.w.errortype.setText(" Size not set")
                self.w.show()

    def download_model(self):   # function to download the model
        try:

            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','Pre_Trained_models',"pickle(*.pkl)")
            pkl_filename = name[0]  # getting the file name
            with open(pkl_filename, 'wb') as file:  # opening the file
                pickle.dump(self.lr, file)  # this will dump the object to a file
            self.user_act.save_file(pkl_filename)   # calling the function to save the file
        except:
                self.w =error_window()
                self.w.errortype.setText("Failed to save the file")
                self.w.show()

    def training(self): # function to train the model
        try:

            classifier= []  # creating an empty list
            imported_as= [] 
            from sklearn.linear_model import LogisticRegression # importing the logistic regression model
            lr= LogisticRegression()    # creating an object of the logistic regression model
            classifier.append('Logistic Regression')    # adding the logistic regression to the list
            imported_as.append('lr')    # adding the logistic regression to the list

            from sklearn.neighbors import KNeighborsClassifier  # importing the KNN model
            knn= KNeighborsClassifier(n_neighbors=1)    # creating an object of the KNN model
            classifier.append(' K Nearest Neighbors')   # adding the KNN to the list
            imported_as.append('knn')   # adding the KNN to the list

            from sklearn.svm import SVC
            svc= SVC()
            classifier.append('Support Vector Machine')
            imported_as.append('svc')

            from sklearn.ensemble import RandomForestClassifier
            rfc= RandomForestClassifier()
            classifier.append('Random Forest')
            imported_as.append('rfc')

            from sklearn.naive_bayes import GaussianNB  # importing the Gaussian Naive Bayes model
            nb= GaussianNB()    # creating an object of the Gaussian Naive Bayes model
            classifier.append('Naive Bayes')    # adding the Gaussian Naive Bayes to the list
            imported_as.append('nb')    # adding the Gaussian Naive Bayes to the list

            from sklearn.tree import DecisionTreeClassifier
            dt= DecisionTreeClassifier()
            classifier.append('Decision Tree')
            imported_as.append('dt')
        except:
                self.w =error_window()
                self.w.errortype.setText("First Split your dataset!")
                self.w.show()


        class Modelling:    # creating a class to store the model
            def __init__(self, x_train, y_train, x_test, y_test, models):
                self.x_train= x_train   # storing the train data
                self.x_test= x_test   # storing the test data
                self.y_train= y_train   
                self.y_test= y_test 
                self.models= models
            
            def fit(self):  # function to fit the model
                model_acc= []   # creating an empty list
                model_time= []  
                for i in self.models:       # looping through the models
                    start= time.time()  # starting the timer
                    if i == 'knn':  # if the model is KNN
                        accuracy= []    # creating an empty list
                        for j in range(1, 200): # looping through the values of K
                            kn= KNeighborsClassifier(n_neighbors= j)     # creating an object of the KNN model
                            kn.fit(self.x_train, self.y_train)  
                            predK= kn.predict(self.x_test)  # predicting the test data
                            accuracy.append([metrics.accuracy_score(self.y_test, predK), j])
                        temp= accuracy[0]   # storing the accuracy and the value of K
                        for m in accuracy:  # looping through the accuracy values
                            if temp[0] < m[0]:      # if the accuracy is greater than the previous one
                                temp= m  # storing the new accuracy and the value of K

                        i= KNeighborsClassifier(n_neighbors=temp[1])
                    i.fit(self.x_train, self.y_train)   # fitting the model
                    model_acc.append(metrics.accuracy_score(self.y_test, i.predict(self.x_test)))   # calculating the accuracy of the model
                    stop= time.time()   # stopping the timer
                    model_time.append((stop-start))  # calculating the time taken to train the model
                self.models_output= pd.DataFrame({'Models': self.models, 'Accuracy': model_acc, 'Runtime (s)': model_time})
            
            def results(self):
                models= self.models_output  # storing the model output
                models= models.sort_values(by= ['Accuracy', 'Runtime (s)'], ascending=[False, True]).reset_index().drop('index', axis=1)
                self.best= models['Models'][0]  # storing the best model
                models['Models']= models['Models'].astype(str).str.split("(", n=2, expand= True)[0]
                models['Accuracy']= models['Accuracy'].round(5)*100 # rounding the accuracy to 5 decimal places
                self.models_output_cleaned= models  # storing the cleaned model output
                return (models)
            
            def best_model(self, type): # function to return the best model
                if type== 'model':  # if the type is model
                    return(self.best)   # returning the best model
                elif type== 'name': # if the type is name
                    return(self.models_output_cleaned['Models'][0])
        
            def best_model_accuracy(self):  # function to return the best model accuracy
                return(self.models_output_cleaned['Accuracy'][0])   
            
            def best_model_runtime(self):   # function to return the best model runtime
                return(round(self.models_output_cleaned['Runtime (s)'][0], 3))
                
            def best_model_confusion_matrix(self):  # function to return the confusion matrix of the best model
                return(metrics.confusion_matrix(self.y_test, self.best.predict(self.x_test)))   # returning the confusion matrix of the best model

            def best_model_clmatrix(self):  # function to return the classification matrix of the best model
                return(metrics.classification_report(self.y_test, self.best.predict(self.x_test)))  # returning the classification matrix of the best mode

            def best_predict(self, x_testp):    # function to predict the test data
                return self.best.predict(x_testp)   # returning the prediction of the test data

        self.models_to_test= [rfc, lr, knn, svc, nb, dt]    # storing the models to test
        self.classification= Modelling(self.x_train, self.y_train, self.x_test, self.y_test, self.models_to_test)
        self.classification.fit()   # fitting the models
        self.classification.results()   # returning the results of the models
        
        
        self.y_pred= self.classification.best_predict(self.x_test)  # predicting the test data
        self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.y_pred))) # displaying the mean absolute error
        self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.y_pred)))  # displaying the mean squared error
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.y_pred))))    # displaying the root mean squared error
        self.acc_sc.setText(str(metrics.accuracy_score(self.y_test, self.y_pred)))  # displaying the accuracy score
        text=steps.classification_(self.y_test,self.y_pred) # returning the classification report
        self.report.setPlainText(text)  # displaying the classification report

    def set_predict(self):  # function to predict the test data
        # try:

        self.a = self.list.text()   # storing the value of the list
        self.ls = self.a.split(",") # splitting the value of the list
        self.target.setText(str(self.target_value))
        self.ls_updated = [float(x) for x in self.ls]   # converting the values of the list to float
        self.ls_array =  np.array(self.ls_updated)  # converting the values of the list to array
        self.pred = self.classification.best_predict([self.ls_array])   # predicting the test data
        # print(self.df)
        
        self.predict_val.setText(str(self.pred))    # displaying the prediction
        a = str(self.target_value)
        # print(a)
        self.df2 = pd.DataFrame(self.df_original[a].copy())
        self.df_original[a],func_name =data.convert_category(self.df_original,a)
        print(self.df_original[a])
        # print(self.df2)
        # print(self.df2[a])
        self.dict_val = dict(zip(self.df_original[a],self.df2.iloc[:,0]))
        
        print(self.dict_val)    # printing the dictionary
        pred_str = str(self.pred).replace('[','').replace(']','')
        print(self.dict_val[float(pred_str)])

        # self.dict_val  = UI.decode()    # decoding the values
        # except:
        #   self.w =error_window()
        #   self.w.errortype.setText("Error! Try Again")
        #   self.w.show()
        
    
    

    def plt3d(self):    # function to plot the 3D graph
        try:
                
            pio.renderers.default= 'browser'    # setting the default renderer to browser
            fig= px.scatter_3d(data_frame= self.df, x= self.X_combo.currentText(), y=self.Y_combo.currentText(), z=self.Z_combo.currentText(), color=self.color_combo.currentText())
            return(pio.show(fig))   # returning the 3D graph
        except:
                self.w =error_window()
                self.w.errortype.setText("Columns not selected")
                self.w.show()

    def conf_matrix(self):  # function to return the confusion matrix

        data = {'y_Actual':self.y_test.values,'y_Predicted':self.y_pred }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted']) # creating a dataframe
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()    # creating a figure
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()  # displaying the confusion matrix
