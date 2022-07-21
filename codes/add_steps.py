
from sklearn.pipeline import Pipeline  

class add_steps:	
	
	def __init__(self):

		self.text=""	# text to be saved in a file
		self.code=""

	# def delete_text(self):

	# 	self.text=""

	def add_text(self,text):	# text to be saved in a file

		self.text=self.text + "\n" + text	# text to be saved in a file
		#print(self.text)	# text to be saved in a file
		#print(text+" added")

	def save_file(self,filename):	# text to be saved in a file

	
		filename = filename.split(".")	
		filename=filename[0]+".txt"
		f=open(filename, 'w')
		print(self.text)
		f.write(self.text)

	def add_pipeline(self,name,action):	# 

		print(name,"\n")
		print(action)

	def add_code(self,text):

		self.code=self.code+"\n"+text
		print(self.code)