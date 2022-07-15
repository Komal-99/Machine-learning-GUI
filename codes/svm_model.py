
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QTextEdit ,QListWidget ,QTableView ,QComboBox,QLabel,QLineEdit,QTextBrowser
import sys,pickle
import data_visualise
import table_display
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
		# self.test_size= self.findChild(QLabel,"test_size") 
	  
		self.c_=self.findChild(QLineEdit,"c_")
		self.random=self.findChild(QLineEdit,"randomstate")
		self.kernel=self.findChild(QComboBox,"kernel")
		self.degree=self.findChild(QLineEdit,"degree")
		self.gamma=self.findChild(QComboBox,"gamma")
		# self.custom_gamma=self.findChild(QLineEdit,"custom_gamma")

		self.coef=self.findChild(QLineEdit,"coef")
		self.max_iter=self.findChild(QLineEdit,"max_iter")
		self.dec_func=self.findChild(QComboBox,"dec_func")  
		self.train_btn=self.findChild(QPushButton,"train")
		self.mae=self.findChild(QLabel,"mae")
		self.mse=self.findChild(QLabel,"mse")
		self.rmse=self.findChild(QLabel,"rmse")
		# self.roc_btn=self.findChild(QPushButton,"roc")
		# self.X_combo=self.findChild(QComboBox,"X_combo")
		# self.Y_combo=self.findChild(QComboBox,"Y_combo")

		self.test_data=self.findChild(QLineEdit,"test_data")
		self.test_size_btn=self.findChild(QPushButton,"test_size_btn")
		self.train_btn.clicked.connect(self.training)
		self.conf_mat_btn=self.findChild(QPushButton,"conf_mat")
		# self.roc_btn=self.findChild(QPushButton,"visualize")
		# self.roc_btn.clicked.connect(self.roc_plot)
		self.conf_mat_btn.clicked.connect(self.conf_matrix)
		self.test_size_btn.clicked.connect(self.test_split)
		self.dwnld.clicked.connect(self.download_model)
		self.list=self.findChild(QLineEdit,"list")
		self.predict_btn=self.findChild(QPushButton,"predict")
		self.decission_rgn_btn=self.findChild(QPushButton,"plot")
		self.predict_val =self.findChild(QLabel,"predict_val")
		self.decission_rgn_btn.clicked.connect(self.decision_regions)
		self.predict_btn.clicked.connect(self.set_predict)
		self.setvalue()
		self.show()

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
			pickle.dump(self.svc_model, file)  
		
		self.user_act.save_file(pkl_filename)

	def set_predict(self):
		self.a = self.list.text()
		self.ls = self.a.split(",")
		self.ls_updated = [float(x) for x in self.ls]
		self.ls_array =  np.array(self.ls_updated)
		self.pred  =self.svc_model.predict([self.ls_array])
		self.predict_val.setText(str(self.pred))


	def test_split(self):

		self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.df,self.X[self.target_value],test_size=float(self.test_data.text()),random_state=int(self.random.text()))
		print(self.y_train.shape)
		print(self.y_test.shape)
		# self.train_size.setText(str(self.x_train.shape))
		# self.test_size.setText(str(self.x_test.shape))
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
	# def roc_plot(self):
    # 		self.fpr = dict()
	# 		self.tpr = dict()
	# 		self.roc_auc = dict()
	# 		self.y_score = self.svc_model.fit(self.x_train, self.y_train).decision_function(self.x_test)
	# 		self.y_score = self.svc_model.predict_proba(self.x_test)
	# 		self.y=self.X[self.target_value]
	# 		for i in range(self.n_classes):
				
	# 			self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test[:, i],self.y_score[:, i])
	# 			self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])
	# 		# First aggregate all false positive rates
	# 		all_fpr = np.unique(np.concatenate([self.fpr[i] for i in range(self.n_classes)]))

	# 		# Then interpolate all ROC curves at this points
	# 		mean_tpr = np.zeros_like(all_fpr)
	# 		for i in range(self.n_classes):
	# 			mean_tpr += np.interp(all_fpr, self.fpr[i], self.tpr[i])

	# 		# Finally average it and compute AUC
	# 		mean_tpr /= self.n_classes

	# 		self.fpr["macro"] = all_fpr
	# 		self.tpr["macro"] = mean_tpr
	# 		self.roc_auc["macro"] = auc(self.fpr["macro"], self.tpr["macro"])

	# 		# Plot all ROC curves
	# 		plt.figure()
	# 		plt.plot(
	# 			self.fpr["micro"],
	# 			self.tpr["micro"],
	# 			label="micro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["micro"]),
	# 			color="deeppink",
	# 			linestyle=":",
	# 			linewidth=4,
	# 		)

	# 		plt.plot(
	# 			self.fpr["macro"],
	# 			self.tpr["macro"],
	# 			label="macro-average ROC curve (area = {0:0.2f})".format(self.roc_auc["macro"]),
	# 			color="navy",
	# 			linestyle=":",
	# 			linewidth=4,
	# 		)
	# 		lw = 2
	# 		colors = cycle(["aqua", "darkorange", "cornflowerblue"])
	# 		for i, color in zip(range(self.n_classes), colors):
	# 			plt.plot(
	# 				self.fpr[i],
	# 				self.tpr[i],
	# 				color=color,
	# 				lw=lw,
	# 				label="ROC curve of class {0} (area = {1:0.2f})".format(i, self.roc_auc[i]),
	# 			)

	# 		plt.plot([0, 1], [0, 1], "k--", lw=lw)
	# 		plt.xlim([0.0, 1.0])
	# 		plt.ylim([0.0, 1.05])
	# 		plt.xlabel("False Positive Rate")
	# 		plt.ylabel("True Positive Rate")
	# 		plt.title("Some extension of Receiver operating characteristic to multiclass")
	# 		plt.legend(loc="lower right")
	# 		plt.show()

    