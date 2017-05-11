import matplotlib.pylab as plt
# %matplotlib inline
# import sompy as sompy
import pandas as pd
import numpy as np
from time import time
import sompy
import pickle
from sompy.visualization.bmuhits import BmuHitsView
from Weight_file import Generate_file
import os

def final_run(finaldataset, finalmapsize,dataset_name,type_of_problem="realWorld",final_rough_train=1600,final_fine_train=400):
	# dlen = 200
		# df = pd.read_csv('./iris.csv')
		# #Training inputs for RGBcolors
		# colors =  np.array( [df['Sepal.Length'] , df['Sepal.Width'] , df['Petal.Length'] , df['Petal.Width'] ]).T
		# color_names = ["."]*50 + ["#"]*50 + ["$"]*50
		# dlen = colors.shape[0]
		# Data1=colors
	print finaldataset.shape
	print finalmapsize
	Data1 = finaldataset
	dlen = Data1.shape[0]
	# Data1 = pd.DataFrame(data= 1*np.random.rand(dlen,2))
	# Data1.values[:,1] = (Data1.values[:,0][:,np.newaxis] + .42*np.random.rand(dlen,1))[:,0]


	# Data2 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+1)
	# Data2.values[:,1] = (-1*Data2.values[:,0][:,np.newaxis] + .62*np.random.rand(dlen,1))[:,0]

	# Data3 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+2)
	# Data3.values[:,1] = (.5*Data3.values[:,0][:,np.newaxis] + 1*np.random.rand(dlen,1))[:,0]


	# Data4 = pd.DataFrame(data= 1*np.random.rand(dlen,2)+3.5)
	# Data4.values[:,1] = (-.1*Data4.values[:,0][:,np.newaxis] + .5*np.random.rand(dlen,1))[:,0]


	# Data1 = np.concatenate((Data1,Data2,Data3,Data4))
	# print Data1.shape

	mapsize = finalmapsize #[95,33]
	som = sompy.SOMFactory.build(Data1, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods

	som.train(n_job=1, verbose=None, train_rough_len=final_rough_train, train_finetune_len=final_fine_train)  # verbose='debug' "info"will print more, and verbose=None wont print anything
	
	weight_initial = np.reshape(np.array(som.codebook.matrix_initial),[mapsize[0],mapsize[1],len(som.codebook.matrix_initial[0]) ])
	weight_final = np.reshape(np.array(som.codebook.matrix),[mapsize[0],mapsize[1],len(som.codebook.matrix[0]) ])

	if type_of_problem!="realWorld":
		g1 = Generate_file(weight_initial,os.path.join(dataset_name,dataset_name + "_weight_initial"))
		g1.generate()

		g2 = Generate_file(weight_final,os.path.join(dataset_name,dataset_name + "_weight_final"))
		g2.generate()

	quantization_error = np.mean(som._bmu[1])
	topographic_error = som.calculate_topographic_error()
	print "Optimal Map Size: " , mapsize
	print "Optimal AQE: " , quantization_error 
	print "Optimal TE: ", topographic_error
	print "Optimal Mean of AQE and TE: ", (quantization_error+topographic_error)/2.0
	print "Density: ", ((dlen)/(mapsize[0]*mapsize[1]*1.0))

	vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
	final = vhts.show(som, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

	file1 = open(os.path.join(dataset_name,dataset_name+"_results.txt"),"w")
	file1.write("map size - ")
	file1.write('{}'.format(mapsize))
	file1.write('\n')

	file1.write("quantization error - ")
	file1.write('{}'.format(quantization_error))
	file1.write('\n')

	file1.write("topographic error - ")
	file1.write('{}'.format(topographic_error))
	file1.write('\n')

	file1.write("topographic error and quantization error average - ")
	file1.write('{}'.format((quantization_error+topographic_error)/2.0))
	file1.write('\n')

	file1.write("density - ")
	file1.write('{}'.format((dlen)/(mapsize[0]*mapsize[1]*1.0)))
	file1.write('\n')

	file1.write("HIT map - ")
	file1.write('\n')
	for row in range(final.shape[0]):
		for column in range(final.shape[1]):
			file1.write('{}'.format(final[row][column]) )
			file1.write(" ")
		file1.write('\n')
	file1.write('\n')
	
	return (quantization_error+topographic_error)/2.0
	
#	with open('hitmap.txt', 'wb') as f:
#		pickle.dump(final, f)

	# vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
	# vhts.show(som, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)



	# print som.codebook.matrix
	# print np.array(som.codebook.matrix).shape


	# print som.location_matrix
	#som.train(n_job=1, verbose='None',  train_rough_len=1)  # verbose='debug' will print more, and verbose=None wont print anything
	#print np.mean(som._bmu[1])
	#print som.calculate_topographic_error()

	# with open('matrix.pkl', 'wb') as f:
	#    pickle.dump(som.location_matrix, f)
