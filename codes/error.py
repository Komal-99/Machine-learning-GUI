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