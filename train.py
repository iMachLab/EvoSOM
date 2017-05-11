import pandas as pd
import numpy as np
from evolve import train
from math import *

filepath = './dataset/syntheticdataforexperimentsom/gingerBreadman.txt'

if 'glass.data' in filepath:
	number_of_columns_csv = 11
	features = 9
	density = 0.5
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)])
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	data =  np.array( [ df[str(i)] for i in range(1,features+1) ]).T
	print data.shape
	train(dataset_name="glass_result_dhalf_new",no_generation=10,population_size=5,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="realWorld",
	final_rough_train=1600,final_fine_train=400)

if 'car.data' in filepath:
	number_of_columns_csv = 7
	features = 6
	density = 0.5
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)])
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="car_result_dhalf_new",no_generation=10,population_size=5,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="realWorld",
	final_rough_train=1600,final_fine_train=400)

if 'iris.data' in filepath:
	number_of_columns_csv = 5
	features = 4
	density = 0.5
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)])
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="iris_result_dhalf_new",no_generation=10,population_size=5,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="realWorld",
	final_rough_train=1600,final_fine_train=400)


if 'abalone.data' in filepath:
	number_of_columns_csv = 9
	features = 8
	density = 0.5
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)])
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="abalone_result_dhalf_new",no_generation=10,population_size=5,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="realWorld",
	final_rough_train=1600,final_fine_train=400)

if 'sonar-uniquestring.all-data' in filepath:
	number_of_columns_csv = 61
	features = 60
	density = 0.5
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)])
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="sonar_result_dhalf_new",no_generation=10,population_size=5,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="realWorld",
	final_rough_train=1600,final_fine_train=400)
						
if 'wine.data' in filepath:
	number_of_columns_csv = 14
	features = 14
	density = 0.5
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)])
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	data =  np.array( [ df[str(i)] for i in range(1,features) ]).T
	print data.shape
	train(dataset_name="wine_result_dhalf_new",no_generation=10,population_size=5,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="realWorld",
	final_rough_train=1600,final_fine_train=400)

if 'corner.txt' in filepath:
	number_of_columns_csv = 3
	features = 2
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)], sep='\t')
	density = 2
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	# rangemax=20
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="corner_dtwo_result",no_generation=15,population_size=10,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="syntheticWorld",
	final_rough_train=1600,final_fine_train=400)

if 'crescentFullmoon.txt' in filepath:
	number_of_columns_csv = 3
	features = 2
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)], sep='\t')
	density = 2
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	# rangemax=20
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="crescentFullmoon_dtwo_result",no_generation=15,population_size=10,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="syntheticWorld",
	final_rough_train=1600,final_fine_train=400)

if 'gingerBreadman.txt' in filepath:
	number_of_columns_csv = 3
	features = 2
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)], sep='\t')
	# density = 2
	# dlen = df.shape[0]
	# rangemax = int(ceil(sqrt(dlen/density)))
	rangemax=20
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="gingerBreadman_result",no_generation=15,population_size=10,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="syntheticWorld",
	final_rough_train=1600,final_fine_train=400)

if 'half_kernel.txt' in filepath:
	number_of_columns_csv = 3
	features = 2
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)], sep='\t')
	density = 2
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	# rangemax=20
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="half_kernel_dtwo_result",no_generation=15,population_size=10,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="syntheticWorld",
	final_rough_train=1600,final_fine_train=400)

if 'outliers.txt' in filepath:
	number_of_columns_csv = 3
	features = 2
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)], sep='\t')
	# density = 2
	# dlen = df.shape[0]
	# rangemax = int(ceil(sqrt(dlen/density)))
	rangemax=20
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="outliers_result",no_generation=15,population_size=10,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="syntheticWorld",
	final_rough_train=1600,final_fine_train=400)

if 'two_spirals.txt' in filepath:
	number_of_columns_csv = 3
	features = 2
	df = pd.read_csv(filepath,names=[str(i) for i in range(number_of_columns_csv)], sep='\t')
	density = 2
	dlen = df.shape[0]
	rangemax = int(ceil(sqrt(dlen/density)))
	# rangemax=20
	data =  np.array( [ df[str(i)] for i in range(features) ]).T
	print data.shape
	train(dataset_name="two_spirals_dtwo_result",no_generation=15,population_size=10,
	dim_rangemin=1,dim_rangemax=rangemax,data1=data,len_train_rough1=800,len_train_finetune1=200,type_of_problem="syntheticWorld",
	final_rough_train=1600,final_fine_train=400)
