import numpy as np
import os

## 	NOTE - som.codebook.matrix is 2D matrix (n*m,feature) we need to reshape it to 3D matrix ((n,m,feature)) using n,m value before passing it to this class 


## check test.py for reference or 
## example
# from Weight_file import Generate_file
# m = [[[1, 2], [2, 3]], [[3, 4], [4, 5]], [[5, 6], [6, 7]]]
# g = Generate_file(m,"test")
# g.generate()
# files availbale in test folder


class Generate_file:
	"Used to generate weight files from given 3D matrix"

	# pass the 3D matrix into the constructor along with the folder in which you want to keep all the files
	def __init__(self,matrix,folder_name):
		self.matrix = np.array(matrix)
		self.folder_name = folder_name

	def create_folder(self):
		current_directory = os.path.dirname(os.path.abspath(__file__))
		new_folder_path = os.path.join(current_directory,self.folder_name)
		if not os.path.exists(new_folder_path):
			os.makedirs(new_folder_path)
    		print(new_folder_path,"folder created")

	def generate(self):
		self.create_folder()
		current_directory = os.path.dirname(os.path.abspath(__file__))
		file_path = os.path.join(current_directory,self.folder_name)

		row,column,feature_length = self.matrix.shape

		for f in range(feature_length):
			#files name as 1,2,3....feature_length
			file = open(os.path.join(file_path,str(f+1)),"w")
			print("created ",os.path.join(file_path,str(f+1)))
			for r in range(row):
				for c in range(column):
					point = self.matrix[r][c][f]
					file.write('{}'.format(point))
					file.write(" ")
				file.write("\n")
			file.close()
