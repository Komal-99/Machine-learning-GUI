
from re import X
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys ,pickle
import data_visualise
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score , confusion_matrix , roc_curve, roc_auc_score
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import common as common
from PyQt5.QtCore import QCoreApplication

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
    def __init__(self,df_original,df,target,user_actions):
        
        super(UI, self).__init__()
        uic.loadUi("ui_files/LogisticRegression.ui", self)
        self.df_original = df_original
        self.user_act=user_actions
        global data ,steps
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        self.columns= self.findChild(QListWidget,"columns") 
        self.solver=self.findChild(QComboBox,"solver")        
        self.random=self.findChild(QLineEdit,"randomstate")     
        self.max_iter=self.findChild(QLineEdit,"max_iter")  
        self.multi_class=self.findChild(QComboBox,"multi_class")
        self.train_btn=self.findChild(QPushButton,"train")
        
        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mae_2")
        self.rmse=self.findChild(QLabel,"rmse")
        self.roc_btn=self.findChild(QPushButton,"output")
        self.accuracy=self.findChild(QLabel,"accuracy_score")


        self.exitbutton = self.findChild(QPushButton,"ExitButton")

        self.exitbutton.clicked.connect(self.exit)
        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        self.roc_btn.clicked.connect(self.roc_plot)
        self.conf_mat_btn.clicked.connect(self.conf_matrix)
        self.test_size_btn.clicked.connect(self.test_split)
        self.dwnld_2.clicked.connect(self.download_model)
        self.train_btn.setStyleSheet(
                             "QPushButton::pressed"
                             "{"
                             "background-color : green;"
                             "}"
                             )
        self.split_done= self.findChild(QLabel,"split")
        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
       
        self.predict_btn.clicked.connect(self.set_predict)
        
        self.setvalue()
        self.show()
    def exit(self):
        sys.exit()
    def setvalue(self):
        self.columns.addItems(self.column_list)

    def test_split(self):

        try:

            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
            print(self.y_train.shape)
            print(self.y_test.shape)
            self.split_done.setText(str("Split Done"))

        except:

            self.w =error_window()
            self.w.errortype.setText("Size Not set")
            self.w.show()

    def download_model(self):

        try:

            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','',"pickle(*.pkl)")
        
            pkl_filename = name[0]
            with open(pkl_filename, 'wb') as file:
                model= (self.lr, accuracy_score(self.pre,self.y_test))
                pickle.dump(model, file)  # this will dump the object to a file
            
            self.user_act.save_file(pkl_filename)  

        except:

            self.w = error_window()
            self.w.errortype.setText("Failed to save the file")
            self.w.show()

    def set_predict(self):
        self.a = self.list.text() 
        self.ls = self.a.split(",")
        self.target.setText(str(self.target_value))
        self.ls_updated = [float(x) for x in self.ls]
        self.ls_array =  np.array(self.ls_updated)
        self.pred  =self.lr.predict([self.ls_array])
        self.predict_val.setText(str(self.pred))
        
        

    def training(self):

        try:

            self.lr = LogisticRegression(max_iter=int(self.max_iter.text()),random_state=1,solver=self.solver.currentText(),multi_class=self.multi_class.currentText())
            self.lr.fit(self.x_train,self.y_train)
            
            self.pre=self.lr.predict(self.x_test)
            self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))
            self.mae_2.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))
            self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))
            self.accuracy_score.setText(str(accuracy_score(self.pre,self.y_test)))
            self.text=steps.classification_(self.y_test,self.pre)
            self.report.setPlainText(self.text)
            # self.accuracy_score.setText(str(self.lr.score(self.x_test,self.y_test)))

        except:

            self.w =error_window()
            self.w.errortype.setText("First Split your dataset!")
            self.w.show()


    def conf_matrix(self):

        try:

            data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }
            df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
            confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
            plt.figure()
            sns.heatmap(confusion_matrix, annot=True)
            plt.show()

        except:

            self.w =error_window()
            self.w.errortype.setText("Train Your Model First!")
            self.w.show() 

    
    def roc_plot(self):

        try:

            self.pre=self.lr.predict(self.x_test)
            self.auc=roc_auc_score(self.y_test,self.pre)
            self.fpr,self.tpr,threshold =roc_curve(self.y_test,self.pre)
            plt.plot(self.fpr,self.tpr,color='red',label='ROC')
            plt.plot([0,1],[0,1],color='darkblue', linestyle='--',label='ROC Curve( area=%0.2f)' %self.auc)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title('Receiver Operating Characteristics Curve')
            plt.legend()
            plt.show()

        except:

            self.w =error_window()
            self.w.errortype.setText("Train Your Model First!")
            self.w.show()