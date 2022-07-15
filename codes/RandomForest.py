
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys,pickle
import data_visualise
from itertools import cycle

import table_display
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.metrics import accuracy_score , confusion_matrix , roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
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
        # self.target = self.findChild(QLabel,"target")
        self.columns= self.findChild(QListWidget,"columns")
        # self.test_size= self.findChild(QLabel,"test_size")  
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
        # self.roc_btn=self.findChild(QPushButton,"roc")
        # self.X_combo=self.findChild(QComboBox,"X_combo")
        # self.Y_combo=self.findChild(QComboBox,"Y_combo")

        self.test_data=self.findChild(QLineEdit,"test_data")
        self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
        self.train_btn.clicked.connect(self.training)
        self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
        self.list=self.findChild(QLineEdit,"list")
        self.predict_btn=self.findChild(QPushButton,"predict")
        self.predict_val =self.findChild(QLabel,"predict_val")
        self.predict_btn.clicked.connect(self.set_predict)

        # self.roc_btn.clicked.connect(self.roc_plot)
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
        # self.target.setText(self.target_value)
        # self.columns.clear()
        self.columns.addItems(self.column_list)
        # self.X_combo.addItems(self.column_list)
        # self.Y_combo.addItems(self.column_list)

    def download_model(self):

        name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
        #file = open(name[0],'w')
        
        pkl_filename = name[0]
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.lr, file)  
        
        self.user_act.save_file(pkl_filename)  
    
    def test_split(self):

        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
        print(self.y_train.shape)
        print(self.y_test.shape)
        # self.train_size.setText(str(self.x_train.shape))
        # self.test_size.setText(str(self.x_test.shape))

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


    
    # def roc_plot(self):
    #     self.pre=self.lr.predict(self.x_test)
    #     self.auc=roc_auc_score(self.y_test,self.pre)
    #     self.fpr,self.tpr,threshold =roc_curve(self.y_test,self.pre)
    #     plt.plot(self.fpr,self.tpr,color='red',label='ROC')
    #     plt.plot([0,1],[0,1],color='darkblue', linestyle='--',label='ROC Curve( area=%0.2f)' %self.auc)
    #     plt.xlabel("FPR")
    #     plt.ylabel("TPR")
    #     plt.title('Receiver Operating Characteristics Curve')
    #     plt.legend()
    #     plt.show()
        # self.fpr = dict()
        # self.tpr = dict()
        # self.roc_auc = dict()
        # # self.y_score = self.lr.fit(self.x_train, self.y_train).decision_function(self.x_test)
        # self.y_score = self.lr.predict_proba(self.x_test)
        # self.y=self.X[self.target_value]
        # for i in range(self.n_classes):
            
        #     self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test[:, i],self.y_score[:, i])
        #     self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        # # First aggregate all false positive rates
        # all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))

        # # Then interpolate all ROC curves at this points
        # mean_tpr = np.zeros_like(all_fpr)
        # for i in range(self.n_classes):
        #     mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])

        # # Finally average it and compute AUC
        # mean_tpr /= self.n_classes

        # self.fpr["macro"] = all_fpr
        # self.tpr["macro"] = mean_tpr
        # self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])

        # # Plot all ROC curves
        # plt.figure()
        # plt.plot(
        #     self.fpr["micro"],
        #     self.tpr["micro"],
        #     label="micro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["micro"]),
        #     color="deeppink",
        #     linestyle=":",
        #     linewidth=4,
        # )

        # plt.plot(
        #     self.fpr["macro"],
        #     self.tpr["macro"],
        #     label="macro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["macro"]),
        #     color="navy",
        #     linestyle=":",
        #     linewidth=4,
        # )
        # lw = 2
        # colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        # for i, color in zip(range(self.n_classes), colors):
        #     plt.plot(
        #         self.fpr[i],
        #         self.tpr[i],
        #         color=color,
        #         lw=lw,
        #         label="ROC curve of class {0} (area = {1:0.2f})".format(i, self.roc_auc[i]),
        #     )

        # plt.plot([0, 1], [0, 1], "k--", lw=lw)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel("False Positive Rate")
        # plt.ylabel("True Positive Rate")
        # plt.title("Some extension of Receiver operating characteristic to multiclass")
        # plt.legend(loc="lower right")
        # plt.show()
        self.fpr = dict()
        self.tpr = dict()
        self.roc_auc = dict()
        # self.y_score = self.lr.fit(self.x_train, self.y_train).decision_function(self.x_test)
        # self.y_score = self.lr.predict_proba(self.x_test)
        self.y=self.X[self.target_value]
        for i in range(self.n_classes):
            self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test[:, i],self.y_score[:, i])
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.n_classes):
            mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= self.n_classes

        self.fpr["macro"] = all_fpr
        self.tpr["macro"] = mean_tpr
        self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(
            self.fpr["micro"],
            self.tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["micro"]),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot(
            self.fpr["macro"],
            self.tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["macro"]),
            color="navy",
            linestyle=":",
            linewidth=4,
        )
        lw = 2
        colors = cycle(["aqua", "darkorange", "cornflowerblue"])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(
                self.fpr[i],
                self.tpr[i],
                color=color,
                lw=lw,
                label="ROC curve of class {0} (area = {1:0.2f})".format(i, self.roc_auc[i]),
            )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Some extension of Receiver operating characteristic to multiclass")
        plt.legend(loc="lower right")
        plt.show()

    