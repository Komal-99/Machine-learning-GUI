
#importing Libraries

from PyQt5.QtWidgets import *
import sys,pickle
from PyQt5 import uic, QtWidgets ,QtCore, QtGui
from sklearn.preprocessing import LabelEncoder
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QDialog,QLineEdit,QLabel
from PyQt5 import QtWidgets
import linear_reg,svm_model,table_display,data_visualise,logistic_reg,RandomForest
import KNN,pre_trained,add_steps, pred_mtnc
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.uic import loadUi
from PyQt5.QtCore import QCoreApplication
import plotly.express as px 
import plotly.io as pio
import plotly.graph_objects as go


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



class home_screen(QDialog):
    def __init__(self):           #initialising the home screen
        super(home_screen,self).__init__()
        loadUi("ui_files/Front Page.ui",self)       #loading the ui file

        #defining the buttons and their functions
        self.Start = self.findChild(QPushButton, "pushButton")
        self.Start.clicked.connect(self.StartButton)
        self.help = self.findChild(QPushButton,"pushButton_3")
        self.help.clicked.connect(self.helpButton)

        
    def helpButton(self):   #help button function
        try:
            
            help = help_screen() #creating an object of help screen
            widget.addWidget(help) #adding the help screen to the widget
            widget.setCurrentIndex(widget.currentIndex()+1)     #setting the current index to the next widget
        except:
                self.w =error_window()
                self.w.errortype.setText(" Some Error Occured Try Again")
                self.w.show()

    def StartButton(self):  #start button function
        self.w =model()     #creating an object of model
        self.w.show()    #showing the model

class model(QMainWindow):
    def __init__(self):
        super(model,self).__init__()
        loadUi("ui_files/modeltype.ui",self)      #loading the ui file
        
        #defining the buttons and their functions for model ui
        self.New_model = self.findChild(QPushButton, "newmodel")
        self.New_model.clicked.connect(self.new)

        self.Trainedmodel = self.findChild(QPushButton,"trainedmodel")
        self.Trainedmodel.clicked.connect(self.train)

        self.exitbutton = self.findChild(QPushButton,"ExitButton")
        # self.exitbutton.clicked.connect(app.instance().quit) #exit button function
        self.exitbutton.clicked.connect(self.exit)
    def exit(self):
        sys.exit()
    def new(self):  #new model button function
        pred = UI() #creating an object of UI
        widget.addWidget(pred)
        widget.setCurrentIndex(widget.currentIndex()+1)
        self.close()    #closing the model type window
        

    def train(self):    #trained model button function
        self.t=trained()
        self.t.show()
        self.close()

class trained(QMainWindow):     #trained model class
    def __init__(self):
        super(trained,self).__init__()
        self.filePath_pre, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home/akshay/Dekstop',"pkl(*.pkl)")    #opening the file dialog
        with open(self.filePath_pre, 'rb') as file:
            self.pickle_model = pickle.load(file)   #loading the pickle file
        # UI.train_func(self)

        # self.X,self.n_classes,self.target_value,self.df,self.column_list=steps.return_data()
        # self.testing=pre_trained.UI(self.df,self.target_value,self.pickle_model,self.filePath_pre)  #creating an object of UI pretrained 
        # loadUi(r"ui_files/pre_trained.ui",self)


 #  Help button connection
class help_screen(QDialog): #help screen class
    def __init__(self):
        super(help_screen,self).__init__()
        loadUi(r"ui_files/Help Centre.ui",self)
        # defining the buttons and their functions
        self.Back = self.findChild(QPushButton, "pushButton")
        self.Back.clicked.connect(self.backButton)

        self.Continue = self.findChild(QPushButton,"pushButton_2")
        self.Continue.clicked.connect(self.continueButton)

    def backButton(self): #back button function
        back = home_screen()
        widget.addWidget(back)  #adding the home screen to the widget
        widget.setCurrentIndex(widget.currentIndex()+1)

    def continueButton(self):   #continue button function
        cont = UI()
        widget.addWidget(cont)
        widget.setCurrentIndex(widget.currentIndex()+1)   #setting the current index to the next widget

class UI(QMainWindow):  #UI class for main window which do data processing and cleaning
    def __init__(self): #initialising the UI class
        super(UI, self).__init__()
        uic.loadUi(r'ui_files\Mainwindow.ui', self) #loading the ui file
        global data,steps                         #globalising the data and steps
        data=data_visualise.data_()             #creating an object of data class
        steps=add_steps.add_steps()            #creating an object of steps class

        #defining the buttons and their functions
        self.Browse = self.findChild(QPushButton,"Browse")
        self.Drop_btn = self.findChild(QPushButton,"Drop")
        self.exitbutton = self.findChild(QPushButton,"ExitButton")
        self.exitbutton.clicked.connect(self.exit)   #exit button function
        self.plot_win= self.findChild(QListWidget,"plotwidget")
        self.con_btn = self.findChild(QPushButton,"convert_btn")
        self.columns= self.findChild(QListWidget,"column_list")
      
        self.cat_column=self.findChild(QComboBox,"cat_column")
        self.table = self.findChild(QTableView,"tableView")
        self.dropcolumns=self.findChild(QComboBox,"dropcolumn")
        self.data_shape = self.findChild(QLabel,"shape")
        self.submit_btn = self.findChild(QPushButton,"Submit")
        self.target_col =self.findChild(QLabel,"target_col")
        self.model_select=self.findChild(QComboBox,"model_select")
        self.scatter_x=self.findChild(QComboBox,"scatter_x")
        self.boxscatter_x=self.findChild(QComboBox,"scatter_x_box")
        self.scatter_y=self.findChild(QComboBox,"scatter_y")
        self.boxscatter_y=self.findChild(QComboBox,"scatter_y_box")
        self.scatter_mark=self.findChild(QComboBox,"scatter_mark")
        self.scatter_c=self.findChild(QComboBox,"scatter_c")
        self.scatter_btn = self.findChild(QPushButton,"scatterplot")
        self.boxplot_btn = self.findChild(QPushButton,"boxplot")
        self.hist_column=self.findChild(QComboBox,"hist_column")
        self.hist_column_add=self.findChild(QComboBox,"hist_column_add")
        self.hist_add_btn = self.findChild(QPushButton,"hist_add_btn")
        self.hist_remove_btn = self.findChild(QPushButton,"hist_remove_btn")
        self.histogram_btn = self.findChild(QPushButton,"histogram")
        self.train=self.findChild(QPushButton,"train")
        self.heatmap_btn = self.findChild(QPushButton,"heatmap")
        self.null_column=self.findChild(QComboBox,"null_column")
        self.nullbtn=self.findChild(QPushButton,"null_2")


        self.X_combo=self.findChild(QComboBox,"X_combo")
        self.Y_combo=self.findChild(QComboBox,"Y_combo")
        self.Z_combo=self.findChild(QComboBox,"Z_combo")
        self.color_combo=self.findChild(QComboBox,"color_combo")
        self.plot3d_btn= self.findChild(QPushButton,"visualize")

        self.Browse.clicked.connect(self.getCSV)    #browse button function
        self.Drop_btn.clicked.connect(self.dropc)   #drop button function
        self.scatter_btn.clicked.connect(self.scatter_plot)     #scatter plot button function
        self.hist_add_btn.clicked.connect(self.hist_add_column) #histogram add button function
        self.hist_remove_btn.clicked.connect(self.hist_remove_column)   #histogram remove button function
        self.histogram_btn.clicked.connect(self.histogram_plot) #histogram button function
        self.heatmap_btn.clicked.connect(self.heatmap_gen)  #heatmap button function
        self.boxplot_btn.clicked.connect(self.box_plot)  #heatmap button function
        self.visualize.clicked.connect(self.plt3d)  #3d plot button function
        self.con_btn.clicked.connect(self.con_cat)  #convert to categorical button function encoding 
        self.columns.clicked.connect(self.target)   #target column button function
        self.submit_btn.clicked.connect(self.set_target)    #fix target value  function
        self.train.clicked.connect(self.train_func) #train button function
        self.scale_btn.clicked.connect(self.scale_value)    #scale button function

        self.nullbtn.clicked.connect(self.fillme)   # fill null values button function
   
        self.show() #showing the main window
    


    def exit(self):
        sys.exit()
    def scale_value(self):  #scaling the values
        try:
            if self.scaler.currentText()=='StandardScale':
                self.df,func_name = data.StandardScale(self.df,self.target_value)   #calling the function from data class converting data into standard scale
            elif self.scaler.currentText()=='MinMaxScale':  #calling the function from data class converting data into min max scale
                self.df,func_name = data.MinMaxScale(self.df,self.target_value)
            elif self.scaler.currentText()=='PowerScale':   #calling the function from data class converting data into power scale
                self.df,func_name = data.PowerScale(self.df,self.target_value)
            
            steps.add_text(self.scaler.currentText()+" applied to data")    #adding the text to the steps
            steps.add_pipeline(self.scaler.currentText(),func_name) #adding the pipeline to the steps
            self.filldetails()  #calling the function to fill the details
        except:
                self.w =error_window()
                self.w.errortype.setText("Select a dataset and target value")
                self.w.show()

    def hist_add_column(self):  #call the function to add a column to the histogram
        
        self.hist_column_add.addItem(self.hist_column.currentText())    #adding the column to the histogram
        self.hist_column.removeItem(self.hist_column.findText(self.hist_column.currentText()))  #removing the column from the histogram


    def hist_remove_column(self): #call the function to remove a column from the histogram
        
        self.hist_column.addItem(self.hist_column_add.currentText()) 
        self.hist_column_add.removeItem(self.hist_column_add.findText(self.hist_column_add.currentText()))


    def histogram_plot(self):       #histogram plot function
        try:
            AllItems = [self.hist_column_add.itemText(i) for i in range(self.hist_column_add.count())]
            for i in AllItems:  #iterating through the columns to plot the histogram
                data.plot_histogram(self.df,i)  #calling the function from data class to plot the histogram
                self.graphWidget.setBackground('w') #setting the background color to white
                self.graphWidget.plot(self.df,i)    #plotting the histogram
        except:
                self.w =error_window()
                self.w.errortype.setText("Dataset/Column not selected")
                self.w.show()

    def plt3d(self): #3d plot function   
        try:  
            fig= go.Figure(data= px.scatter_3d(data_frame= self.df, x= self.X_combo.currentText(), y=self.Y_combo.currentText(), z=self.Z_combo.currentText(), color=self.color_combo.currentText())) 
            return(fig.show())
        except:
                self.w =error_window()  #calling the error window function
                self.w.errortype.setText("Dataset/Column not selected")
                self.w.show()

    def heatmap_gen(self):
        try:
            data.plot_heatmap(self.df)  #calling the function from data class to plot the heatmap
        except:
                self.w =error_window()
                self.w.errortype.setText("Dataframe not selected")
                self.w.show()
    def set_target(self):
        try:    #setting the target value
            self.target_value=str(self.item.text()).split()[0]
            steps.add_code("target=data['"+self.target_value+"']")
            self.target_col.setText(self.target_value)
        except:
                self.w =error_window()
                self.w.errortype.setText("Target column not selected")
                self.w.show()

    def target(self):
            self.item=self.columns.currentItem()
        
    def filldetails(self,flag=1):   #function to fill the details
         
        if(flag==0):    #if the flag is 0 then the data is being loaded from the file
            self.df = data.read_file(str(self.filePath))    #calling the function from data class to read the file
            self.df_original = self.df
        
        self.columns.clear()    #clearing the columns
        self.column_list=data.get_column_list(self.df)  #getting the column list
        self.empty_list=data.get_empty_list(self.df)    #getting the empty list
        self.cat_col_list=data.get_cat(self.df) #getting the categorical list
        for i ,j in enumerate(self.column_list):    #iterating through the column list
            stri=j+ " -------   " + str(self.df[j].dtype)   #getting the column name and the data type
            self.columns.insertItem(i,stri) #inserting the column name and data type to the columns
            

        self.fill_combo_box()   #calling the function to fill the combo box
        shape_df="Shape:  Rows:"+ str(data.get_shape(self.df)[0])+" ,  Columns: "+str(data.get_shape(self.df)[1])
        self.data_shape.setText(shape_df)   #setting the shape of the data to the shape label

    def fill_combo_box(self):   #function to fill the combo box
        
        self.dropcolumns.clear()    #clearing the combo box
        self.dropcolumns.addItems(self.column_list)
        self.cat_column.clear()
        self.cat_column.addItems(self.cat_col_list) #adding the categorical columns to the combo box
        self.scatter_x.clear()
        self.scatter_x.addItems(self.column_list)   #adding the columns to the scatter x combo box
        self.boxscatter_x.clear()
        self.boxscatter_x.addItems(self.column_list)   #adding the columns to the box scatter x combo box
        self.scatter_y.clear()
        self.scatter_y.addItems(self.column_list)   #adding the columns to the scatter y combo box
        self.boxscatter_y.clear()
        self.boxscatter_y.addItems(self.column_list)   #adding the columns to the box scatter y combo box
        self.null_column.clear()
        self.null_column.addItems(self.column_list)  #adding the columns to the null column combo box
       
        self.X_combo.clear()
        self.X_combo.addItems(self.column_list)
        self.Y_combo.clear()
        self.Y_combo.addItems(self.column_list)
        self.Z_combo.clear()
        self.Z_combo.addItems(self.column_list)
        self.color_combo.clear()
        self.color_combo.addItems(self.column_list)
       
        color= ['red', 'green', 'blue', 'yellow']   #list of colors
        self.scatter_c.clear()  #clearing the scatter color combo box
        self.scatter_c.addItems(color)  #adding the colors to the scatter color combo box
        self.hist_column.clear()    
        self.hist_column.addItems(data.get_numeric(self.df))    #adding the numeric columns to the histogram combo box
        self.hist_column.addItem("All") #adding the all option to the histogram combo box
        x=table_display.DataFrameModel(self.df) #creating a dataframe model
        self.table.setModel(x)  #setting the dataframe model to the table
        


    def con_cat(self):  #function to convert categorical columns to numeric
        try: 
            a = str(self.cat_column.currentText())  #getting the categorical column
            self.df2 = self.df[[a]].copy()  #copying the dataframe
            # print(self.df2.iloc[:,0]) 
            self.df[a],func_name =data.convert_category(self.df,a)  #calling the function from data class to convert the categorical column to numeric
            self.dict_val = dict(zip(self.df[a],self.df2.iloc[:,0]))    #creating a dictionary with the categorical values and the numeric values
            steps.add_text("Column "+ a + " converted using Lab elEncoder") #adding the code to the steps
            # print(self.dict_val)
            steps.add_pipeline("LabelEncoder",func_name)    #adding the code to the steps
            self.filldetails()  #calling the function to fill the details

        except:
                self.w =error_window()
                self.w.errortype.setText("dataset not loaded/column not selected")
                self.w.show()

    def decode(self,key):       #function to decode the categorical values
        a = str(self.cat_column.currentText())
        self.df2 = self.df[[a]].copy()      #copying the dataframe
        # print(self.df2.iloc[:,0])
        self.df[a],func_name =data.convert_category(self.df,a)
        self.dict_val = dict(zip(self.df[a],self.df2.iloc[:,0]))    #creating a dictionary with the categorical values and the numeric values
        print(self.dict_val.get(key))   #getting the key from the dictionary
        return self.dict_val



    def fillme(self):   #function to fill the missing values
        try:
            self.df[self.null_column.currentText()]=data.fillmean(self.df,self.null_column.currentText())   #calling the function from data class to fill the missing values
            code="data['"+self.null_column.currentText()+"'].fillna(data['"+self.null_column.currentText()+"'].mean(),inplace=True)"    #creating the code to fill the missing values
            steps.add_code(code)    #adding the code to the steps
            steps.add_text("No Empty Values")   #adding the text to the steps
            self.filldetails()  #calling the function to fill the details
        except:
                self.w =error_window()
                self.w.errortype.setText("String column cannot be filled with mean")
                self.w.show()

    def getCSV(self):   #function to get the csv file
        try:
            self.filePath, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '',"csv(*.csv)")    #getting the file path
            self.columns.clear()    #clearing the columns
            code="data=pd.read_csv('"+str(self.filePath)+"')"   #creating the code to read the csv file
            steps.add_code(code)        #adding the code to the steps
            steps.add_text("File "+self.filePath+" read")   #adding the text to the steps
            if(self.filePath!=""):  #if the file path is not empty
                self.filldetails(0)  #calling the function to fill the details
        except:
                self.w =error_window()
                self.w.errortype.setText("Unable to load file")
                self.w.show()

    def dropc(self):    #function to drop the columns
        try:
            if (self.dropcolumns.currentText() == self.target_value):   #if the target column is selected
                self.target_value=""    #setting the target value to empty
                self.target_col.setText("")  #setting the target column to empty
            self.df=data.drop_columns(self.df,self.dropcolumns.currentText())   #calling the function from data class to drop the columns
            steps.add_code("data=data.drop('"+self.dropcolumns.currentText()+"',axis=1)")   #adding the code to the steps
            steps.add_text("Column "+ self.dropcolumns.currentText()+ " dropped")   #adding the text to the steps
            self.filldetails()  
        except:
                self.w =error_window()
                self.w.errortype.setText("Target Value of dataset not set")
                self.w.show()

    def scatter_plot(self):     #function to create a scatter plot
        try:
            data.scatter_plot(df=self.df,x=self.scatter_x.currentText(),y=self.scatter_y.currentText(),c=self.scatter_c.currentText(),marker=self.scatter_mark.currentText())
        except:
                self.w =error_window()
                self.w.errortype.setText("Arguments not selected")
                self.w.show()


        data.scatter_plot(df=self.df,x=self.scatter_x.currentText(),y=self.scatter_y.currentText(),c=self.scatter_c.currentText(),marker=self.scatter_mark.currentText())
     
    def box_plot(self):     #function to create a box plot
        # data.box_plot(df=self.df,x=self.boxscatter_x.currentText(),y=self.boxscatter_y.currentText())
        try:
            fig = px.box(self.df,
                 x=self.boxscatter_x.currentText(),
                 y=self.boxscatter_y.currentText(),
                #  color  =  "Failure Type",    

                 width  =  800,
                 height =  400)
            fig.show()
        except:
            self.w =error_window()
            self.w.errortype.setText("Columns not selected")
            self.w.show()
        
    def train_func(self):   #function to train the model
        try:

            myDict={ "Linear Regression":linear_reg , "SVM":svm_model , "Logistic Regression":logistic_reg ,"Random Forest":RandomForest,
            "K-Nearest Neighbour":KNN ,"Predictive Maintenace":pred_mtnc}   #creating a dictionary with the model names and the functions

            
            if(self.target_value!=""):  #if the target value is not empty
                
                self.win = myDict[self.model_select.currentText()].UI(self.df_original,self.df,self.target_value,steps)  #calling the function to train the model
        except:
                self.w =error_window()
                self.w.errortype.setText("Select the model")
                self.w.show()

app = QApplication(sys.argv)    #creating an application
welcome = home_screen() #creating an object of the home screen
widget = QtWidgets.QStackedWidget() #creating a stacked widget
widget.addWidget(welcome)   #adding the home screen to the stacked widget
widget.setFixedHeight(920)  #setting the height of the stacked widget
widget.setFixedWidth(1408)  #setting the width of the stacked widget
widget.show()   #showing the stacked widget

# try:    #try block to catch the exception
sys.exit(app.exec_())        #executing the application
# except:     #except block to catch the exception
#     print("exiting..")  #printing the exception
