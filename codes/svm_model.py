
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys,pickle
import data_visualise
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import cycle
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import common


class UI(QMainWindow):
	def __init__(self,df,target,user_actions):
		super(UI, self).__init__()
		uic.loadUi("ui_files/SVM.ui", self)
		self.user_act=user_actions
		global data ,steps
		data=data_visualise.data_()
		steps=common.common_steps(df,target)
		self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
		self.target = self.findChild(QLabel,"target")
		self.columns= self.findChild(QListWidget,"columns")
		
		self.c_=self.findChild(QLineEdit,"c_")
		self.random=self.findChild(QLineEdit,"randomstate")
		self.kernel=self.findChild(QComboBox,"kernel")
		self.degree=self.findChild(QLineEdit,"degree")
		self.gamma=self.findChild(QComboBox,"gamma")
		self.split_done= self.findChild(QLabel,"split")
		self.target=self.findChild(QLabel,"target")
		
		self.coef=self.findChild(QLineEdit,"coef")
		self.max_iter=self.findChild(QLineEdit,"max_iter")
		self.dec_func=self.findChild(QComboBox,"dec_func")  
		self.train_btn=self.findChild(QPushButton,"train")
		self.mae=self.findChild(QLabel,"mae")
		self.mse=self.findChild(QLabel,"mse")
		self.rmse=self.findChild(QLabel,"rmse")

		self.test_data=self.findChild(QLineEdit,"test_data")
		self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
		self.train_btn.clicked.connect(self.training)
		self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
	
		self.conf_mat_btn.clicked.connect(self.conf_matrix)
		self.test_size_btn.clicked.connect(self.test_split)
		self.dwnld.clicked.connect(self.download_model)
		self.list=self.findChild(QLineEdit,"list")
		self.predict_btn=self.findChild(QPushButton,"predict")
		self.decission_rgn_btn=self.findChild(QPushButton,"plot")
		self.predict_val =self.findChild(QLabel,"predict_val")
		self.decission_rgn_btn.clicked.connect(self.decision_regions)
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
		

	def download_model(self):

		name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File','/home/akshay/Desktop',"pickle(*.pkl)")
		pkl_filename = name[0]
		with open(pkl_filename, 'wb') as file:
			pickle.dump(self.svc_model, file)  
		
		self.user_act.save_file(pkl_filename)

	def set_predict(self):
		self.a = self.list.text()
		self.ls = self.a.split(",")
		self.target.setText(str(self.target_value))
		self.ls_updated = [float(x) for x in self.ls]
		self.ls_array =  np.array(self.ls_updated)
		self.pred  =self.svc_model.predict([self.ls_array])
		self.predict_val.setText(str(self.pred))


	def test_split(self):

		self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
		print(self.y_train.shape)
		print(self.y_test.shape)
		self.split_done.setText(str("Split Done"))
		
	def decision_regions(self):
		self.plotting=self.column_list[2:]
		print(self.plotting)
		value=0
		width=0
		plot_decision_regions(X=self.x_train.values,
					  y=self.y_train.values,
					  clf=self.svc_model,
					  filler_feature_values={i+2:value for i,j in enumerate(self.plotting) },
					  filler_feature_ranges={i+2:width for i,j in enumerate(self.plotting)},
					  zoom_factor=0.1,
					  legend=2)
		plt.show()
	def training(self):

		self.svc_model = SVC(C=float(self.c_.text()),kernel=self.kernel.currentText(),degree=float(self.degree.text()),gamma=self.gamma.currentText(),coef0=float(self.coef.text()),decision_function_shape=self.dec_func.currentText(),probability=True)
		self.svc_model.fit(self.x_train.values,self.y_train.values)
		self.pre=self.svc_model.predict(self.x_test)
		self.mae.setText(str(metrics.mean_absolute_error(self.y_test,self.pre)))
		self.mse.setText(str(metrics.mean_squared_error(self.y_test,self.pre)))
		self.rmse.setText(str(np.sqrt(metrics.mean_squared_error(self.y_test,self.pre))))
		text=steps.classification_(self.y_test,self.pre)
		self.report.setPlainText(text)

	def conf_matrix(self):

		data = {'y_Actual':self.y_test.values,'y_Predicted':self.pre }
		df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
		confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
		plt.figure()
		sns.heatmap(confusion_matrix, annot=True)
		plt.show()
	