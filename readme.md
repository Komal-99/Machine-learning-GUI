Run the uicode.py file to start the main window

>> Soon Demo video will be availaible

# Introduction
This introduction provides an overview of the Machine Learning Model through the document. It covers the system's purpose, scope, target audience, main component design, and high-level system design considerations.

# Scope and Purpose
This document describes the technical design of the Machine Learning Model - Using ML and Python library known as pyqt5 which is used for GUI programming and designing. The primary goal of this document is to describe the technical vision for how machine learning can be realized in various ways, as well as for predictive maintenance to save a device from sudden and unexpected wear and tear. This document depicts various aspects of the device through an architectural overview of the system. It can also be used as a starting point for developers.

# Target Audience
This document is intended for technical stakeholders:
•	IT Management 
•	IT Management Support Staff 
•	Development Team
The reader is assumed to have a technical background in software design and development. It can be used by anyone dealing with large amounts of data and go through data visualization and data cleaning.

# Languages and Technology Used
Python, pyqt5, Machine Learning, Predictive Maintenance
Setup and Instructions
1.	Install python 3.7 and above version as per your system  configuration 32/64x bit
2.	Git clone the repository https://github.com/Komal-99/Machine-learning-GUI or unzip the file
3.	Open the project directory in your vs code or any python code compiler
4.	Run the command “pip install -r requirements.txt”
5.	After successful installation of all the packages.
6.	Run python uicode.py

# Architecture Design

The Machine Learning Model has a simple Architecture, which allows the researcher to easily go through a large dataset and prepare the data that they want for further research. Because of its uncomplicated architecture, this application will greatly simplify the process. The relative stability of the components, as seen by the application developer, is a major advantage of this design.

# Modules and Working
Steps to Get Started
Step 1 - Start
Step 2 - In the data processing tab, Browse your dataset.
Step 3 – You can see the set of rows of your dataset in the right top corner and Column description on Left top corner.
Step 4 – In column description select the column as your target. Press the set target button.
Step 5- Pre- Process the data using functionalities like drop column, fill null values, scale your dataset.
Step 6 – You can visualize your data column with the help of different graphs available in the visualization tab like histogram, bar graph, line chart, heatmap, etc. To analyze your dataset.
Step 7 – Select the ML algorithm and train your model OR you can upload your already trained model (.sav) format file to use it for further processes.
Modules and Libraries Used

PyQt5.QtWidgets

In Qt 'widget' is the name given to a component of the UI that the user can interact with. User interfaces are made up of multiple widgets, arranged within the window.

sys,pickle

It's the process of converting a Python object into a byte stream to store it in a file/database, maintain program state across sessions, or transport data over the network. It also works on file handling like loading or saving the file.

QtCore, QtGui

PyQt API is a set of modules containing a large number of classes and functions. While QtCore module contains non-GUI functionality for working with file and directory etc., QtGui module contains all the graphical controls.

QApplication, QWidget, QDialog, QLineEdit, QLabel

These are used for adding the various labels, buttons, textbox, etc. to the widgets or the window.

sklearn 

Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.

LabelEncoder

Sklearn provides a very efficient tool for encoding the levels of categorical features into numeric values. LabelEncoder encode labels with a value between 0 and n_classes-1 where n is the number of distinct labels. If a label repeats it assigns the same value to as assigned earlier.

linear_reg,svm_model,table_display,data_visualise
logistic_reg,RandomForest
KNN,pre_trained,add_steps, pred_mtnc

These are the libraries to import various Machine Learning Algorithms to the program. These include Linear Regression, Logistic Regression, Random Forest, KNN, etc.

matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 

Calls the draw and print fig methods, creates the renderers, etc. Modified to import from Qt5 backend for new-style.

matplotlib.pyplot
plotly.express
plotly.io
plotly.graph_objects
These are the various data visualization libraries for plotting graphs.
These libraries are stored in UI files of the Repository. For the Machine Learning Model to work as per our profit and needs, we connect various ML algorithms (that is mentioned in the later sections) to the UI files through pyqt5 GUI.
