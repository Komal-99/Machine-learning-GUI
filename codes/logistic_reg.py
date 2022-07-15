
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys ,pickle
import data_visualise
import table_display
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
import common



class UI(QMainWindow):
    def __init__(self,df,target,user_actions):
        super(UI, self).__init__()
        uic.loadUi("ui_files/LogisticRegression.ui", self)
        self.user_act=user_actions
        global data ,steps
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        # self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        # self.test_size= self.findChild(QLabel,"test_size") 
        # self.target = self.findChild(QLabel,"target")
        # self.columns= self.findChild(QListWidget,"columns")
        # self.test_size= self.findChild(QLabel,"test_size")  
        
        # self.c_=self.findChild(QLineEdit,"c_")
        # self.penalty=self.findChild(QComboBox,"penalty")
        # self.dual=self.findChild(QComboBox,"dual")   
        self.solver=self.findChild(QComboBox,"solver")        
        self.random=self.findChild(QLineEdit,"randomstate")     
        self.max_iter=self.findChild(QLineEdit,"max_iter")  
        self.multi_class=self.findChild(QComboBox,"multi_class")
        # self.tol=self.findChild(QLineEdit,"tol")
        self.train_btn=self.findChild(QPushButton,"train")
        
        self.mae=self.findChild(QLabel,"mae")
        self.mse=self.findChild(QLabel,"mae_2")
        self.rmse=self.findChild(QLabel,"rmse")
        self.roc_btn=self.findChild(QPushButton,"output")
        self.accuracy=self.findChild(QLabel,"accuracy_score")
        
        # self.X_combo=self.findChild(QComboBox,"X_combo")
        # self.Y_combo=self.findChild(QComboBox,"Y_combo")

        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        self.roc_btn.clicked.connect(self.roc_plot)
        self.conf_mat_btn.clicked.connect(self.conf_matrix)
        self.test_size_btn.clicked.connect(self.test_split)
        self.dwnld_2.clicked.connect(self.download_model)
        
        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_predict)
        
        self.setvalue()
        self.show()

    def setvalue(self):
        # self.target.setText(self.target_value)
        # self.columns.clear()
        self.columns.addItems(self.column_list)
        # self.X_combo.addItems(self.column_list)
        # self.Y_combo.addItems(self.column_list)

   
    def set_predict(self):
        self.a = self.list.text()
        self.ls = self.a.split(",")
       
        self.ls_updated = [float(x) for x in self.ls]
        self.ls_array =  np.array(self.ls_updated)

        self.pred  =self.lr.predict([self.ls_array])
        self.predict_val.setText(int(self.pred))


    
    def test_split(self):

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
        print(self.y_train.shape)
        print(self.y_test.shape)
        # self.train_size.setText(str(self.x_train.shape))
        # self.test_size.setText(str(self.x_test.shape))

    def download_model(self):

        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
        #file = open(name[0],'w')
        
        pkl_filename = name[0]
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.lr, file)  
        
        self.user_act.save_file(pkl_filename)  

    def training(self):

        self.lr = LogisticRegression(max_iter=int(self.max_iter.text()),random_state=1,solver=self.solver.currentText(),multi_class=self.multi_class.currentText())
        self.lr.fit(self.x_train,self.y_train)
        
        self.pre=self.lr.predict(self.x_test)
        self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))
        self.mae_2.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))
        self.accuracy_score.setText(str(accuracy_score(self.pre,self.y_test)))
        self.text=steps.classification_(self.y_test,self.pre)
        self.report.setPlainText(self.text)
        self.accuracy_score.setText(str(self.lr.score(self.x_test,self.y_test)))


    def conf_matrix(self):

        data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

    
    def roc_plot(self):
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