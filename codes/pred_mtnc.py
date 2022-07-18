
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys ,pickle
from PyQt5.QtCore import QCoreApplication
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
import uicode



class UI(QMainWindow):
    def __init__(self,df,target,user_actions):
        super(UI, self).__init__()
        uic.loadUi("ui_files/PredictiveMaintenance.ui", self)
        self.user_act=user_actions
        global data ,steps
        data=data_visualise.data_()
        steps=common.common_steps(df,target)
        self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
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

        self.exitbutton.clicked.connect(QCoreApplication.instance().quit)
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
        
        self.train_btn.clicked.connect(self.training) 
        self.conf_mat_btn.clicked.connect(self.conf_matrix) 
        self.test_size_btn.clicked.connect(self.test_split) 
        self.dwnld.clicked.connect(self.download_model)
        self.visualize.clicked.connect(self.plt3d) 
        #self.acc.clicked.connect(self.accuracy) 

        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_predict)
        self.train_btn.setStyleSheet(
                             "QPushButton::pressed"
                             "{"
                             "background-color : green;"
                             "}"
                             )
        self.setvalue()
        self.show()

    def setvalue(self):
      
        self.columns.addItems(self.column_list)
        # self.column_list= data.get_column_list(self.df)
        self.X_combo.clear()
        self.X_combo.addItems(self.column_list)
        self.Y_combo.clear()
        self.Y_combo.addItems(self.column_list)
        self.Z_combo.clear()
        self.Z_combo.addItems(self.column_list)
        self.color_combo.clear()
        self.color_combo.addItems(self.column_list)

    def test_split(self):

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=0)
        print(self.y_train.shape)
        print(self.y_test.shape)
        self.split_done.setText(str("Split Done"))

    def download_model(self):

        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
        pkl_filename = name[0]
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.lr, file)  # this will dump the object to a file
        self.user_act.save_file(pkl_filename)  

    def training(self):
        classifier= []
        imported_as= []
        from sklearn.linear_model import LogisticRegression
        lr= LogisticRegression()
        classifier.append('Logistic Regression')
        imported_as.append('lr')

        from sklearn.neighbors import KNeighborsClassifier
        knn= KNeighborsClassifier(n_neighbors=1)
        classifier.append(' K Nearest Neighbors')
        imported_as.append('knn')

        from sklearn.svm import SVC
        svc= SVC()
        classifier.append('Support Vector Machine')
        imported_as.append('svc')

        from sklearn.ensemble import RandomForestClassifier
        rfc= RandomForestClassifier()
        classifier.append('Random Forest')
        imported_as.append('rfc')

        from sklearn.naive_bayes import GaussianNB
        nb= GaussianNB()
        classifier.append('Naive Bayes')
        imported_as.append('nb')

        from sklearn.tree import DecisionTreeClassifier
        dt= DecisionTreeClassifier()
        classifier.append('Decision Tree')
        imported_as.append('dt')

        
        class Modelling:
            def __init__(self, x_train, y_train, x_test, y_test, models):
                self.x_train= x_train
                self.x_test= x_test
                self.y_train= y_train
                self.y_test= y_test
                self.models= models
            
            def fit(self):
                model_acc= []
                model_time= []

                for i in self.models:
                    start= time.time()
                    if i == 'knn':
                        accuracy= []
                        for j in range(1, 200):
                            kn= KNeighborsClassifier(n_neighbors= j)
                            kn.fit(self.x_train, self.y_train)
                            predK= kn.predict(self.x_test)
                            accuracy.append([metrics.accuracy_score(self.y_test, predK), j])
                        temp= accuracy[0]
                        for m in accuracy:
                            if temp[0] < m[0]:
                                temp= m
                        i= KNeighborsClassifier(n_neighbors=temp[1])
                    i.fit(self.x_train, self.y_train)
                    model_acc.append(metrics.accuracy_score(self.y_test, i.predict(self.x_test)))
                    stop= time.time()
                    model_time.append((stop-start))
                self.models_output= pd.DataFrame({'Models': self.models, 'Accuracy': model_acc, 'Runtime (s)': model_time})
            
            def results(self):
                models= self.models_output
                models= models.sort_values(by= ['Accuracy', 'Runtime (s)'], ascending=[False, True]).reset_index().drop('index', axis=1)
                self.best= models['Models'][0]
                models['Models']= models['Models'].astype(str).str.split("(", n=2, expand= True)[0]
                models['Accuracy']= models['Accuracy'].round(5)*100
                self.models_output_cleaned= models
                return (models)
            
            def best_model(self, type):
                if type== 'model':
                    return(self.best)
                elif type== 'name':
                    return(self.models_output_cleaned['Models'][0])
        
            def best_model_accuracy(self):
                return(self.models_output_cleaned['Accuracy'][0])
            
            def best_model_runtime(self):
                return(round(self.models_output_cleaned['Runtime (s)'][0], 3))
                
            def best_model_confusion_matrix(self):
                return(metrics.confusion_matrix(self.y_test, self.best.predict(self.x_test)))

            def best_model_clmatrix(self):
                return(metrics.classification_report(self.y_test, self.best.predict(self.x_test)))

            def model_3d_plot(self):
                fig= px.scatter_3d(data, x= 'Air temperature [K]', y='Rotational speed [rpm]', z='Torque [Nm]', color='Failure Type')
                return(fig.show())

            def best_predict(self, x_testp):
                return self.best.predict(x_testp)

        self.models_to_test= [rfc, lr, knn, svc, nb, dt]
        self.classification= Modelling(self.x_train, self.y_train, self.x_test, self.y_test, self.models_to_test)
        self.classification.fit()
        self.classification.results()
        
        
        self.y_pred= self.classification.best_predict(self.x_test)
        self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.y_pred)))
        self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.y_pred)))
        self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.y_pred))))
        self.acc_sc.setText(str(metrics.accuracy_score(self.y_test, self.y_pred)))
        text=steps.classification_(self.y_test,self.y_pred)
        self.report.setPlainText(text)

    def set_predict(self):
        self.a = self.list.text()
        self.ls = self.a.split(",")
        self.target.setText(str(self.target_value))
        self.ls_updated = [float(x) for x in self.ls]
        self.ls_array =  np.array(self.ls_updated) 
        self.pred = self.classification.best_predict([self.ls_array])
        self.predict_val.setText(str(self.pred))
        
        uicode.decode(float(self.pred))
        

    def plt3d(self):
        pio.renderers.default= 'browser'
        fig= px.scatter_3d(data_frame= self.df, x= self.X_combo.currentText(), y=self.Y_combo.currentText(), z=self.Z_combo.currentText(), color=self.color_combo.currentText())
        return(pio.show(fig))

    def conf_matrix(self):

        data = {'y_Actual':self.y_test.values,'y_Predicted':self.y_pred }
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        plt.figure()
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()
