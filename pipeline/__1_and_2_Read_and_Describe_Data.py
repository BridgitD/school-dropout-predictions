"""
Joshua Mausolf - CAPP 30254 Assignment pa3.

Welcome to this relatively consumer friendly Python Script. 
This file contains the source code to run the summary statistics. 
The summary statistics output have been saved in the folder SUMMARY REPORT.


To analyze your data, there are two steps:

	1. 		First, choose your dataset.


	2. 		Second, run the following command in your terminal:

			python __1_and_2_Read_and_Describe_Data.py 

"""


#1	#SET DATASET
	#Examples:
	#dataset = 'data/cs-training.csv' #Original data
	#dataset = 'cs-training#3B.csv' #Post-impute data

dataset = 'data/cohort1_all.csv'




#2 	RUN THIS COMMAND (Do not use quotes or #.)
	
	#  "python __1_and_2_Read_and_Describe_Data.py"





#_______________________________________________________________________________________#
			
			## ******-- SOURCE CODE -- DO NOT MODIFY --***** ##
#_______________________________________________________________________________________#


"""
Below is the source code to run the summary statistics. 
To the consumer: please do not modify. 
"""
import sys, os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


#______________ FUNCTIONS __________________________#

def camel_to_snake(column_name):
    """
    Converts a string that is camelCase into snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


#Draw Histogram Function

def bar(variable, dataset):

	#Define Data
	data = pd.read_csv(dataset, index_col=0, low_memory=False)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#Generate Graph
	fig =data.groupby(variable).size().plot(kind='bar')
	fig.set_xlabel(variable) #defines the x axis label
	fig.set_ylabel('Number of Observations') #defines y axis label
	fig.set_title(variable+' Distribution') #defines graph title
	plt.draw()
	plt.savefig("output/histograms/"+variable+"_bar.jpg")
	plt.close('all')



def histogram1(variable, dataset, color, bins):

	#Define Data
	data = pd.read_csv(dataset, index_col=0, low_memory=False)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#Generate Graph
	fig = data[variable].hist(bins=bins, color=color)
	fig.set_xlabel(variable) #defines the x axis label
	fig.set_ylabel('Number of Observations') #defines y axis label
	fig.set_title(variable+' Distribution') #defines graph title
	plt.draw()
	plt.savefig("output/histograms/"+variable+"_histogram1_"+str(bins)+".jpg")
	plt.clf()



def histogram2(variable, dataset, color, np1, np2):

	#Define Data
	data = pd.read_csv(dataset, index_col=0, low_memory=False)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#Generate Graph
	fig = data[variable].hist(bins=np.arange(np1, np2), color=color)
	fig.set_xlabel(variable) #defines the x axis label
	fig.set_ylabel('Number of Observations') #defines y axis label
	fig.set_title(variable+' Distribution') #defines graph title
	plt.draw()
	plt.savefig("output/histograms/"+variable+"_histogram2_"+str(np1)+".jpg")
	plt.clf()


def line_count(dataset):
		with open(dataset, 'rU') as data_file:
			reader = csv.reader(data_file)
			lines = list(reader)
			#Total File Rows
			XR = len(lines)
			return XR

def dataset_describe(dataset):
	with open(dataset, 'rU') as data_file:
		reader = csv.reader(data_file)
		lines = list(reader)
		#Total File Rows
		XR = len(lines)
		print "Total requested lines:", XR-1

		#Total Number of Variables
		variables = lines[0]
		#print variables[1]
		numVar = len(variables)
		print "Total number of variables: ", numVar
		non_ID_var = variables[1: 12]



def summarize_dataset(dataset):
	"""Select dataset to summarize. Use this function to summarize a dataset.
	To focus on specific variables, please use summary_statistics instead."""

	#Define Data
	data = pd.read_csv(dataset, index_col=0, low_memory=False)
	data.columns = [camel_to_snake(col) for col in data.columns]

	for variable in data.columns:

		print "_"*50
		print "Summary Statistics "+str(variable)+": "
		count = (data[str(variable)].count())
		Number_variable_lines = line_count(dataset)-1
		print "Missing values: ", (Number_variable_lines - count)
		print "Describe "+str(variable)+": ", '\n', (data[str(variable)].describe())
		print "Mode: ", (data[str(variable)].mode())
		#Histogram
		if count > 1:
			histogram1(str(variable), dataset, 'c', 5)
			histogram1(str(variable), dataset, 'g', 10)
			histogram2(str(variable), dataset, 'b', 1.5, 10)
			histogram2(str(variable), dataset, 'r', 1, 10)



def summary_statistics(variable, dataset, bin1=5, bin2=10):
	"""Select variable to summarize. Please input the dataset.
		Histogram bins can be modified. Default is 5 and 10."""

	#Define Data
	data = pd.read_csv(dataset, index_col=0, low_memory=False)
	data.columns = [camel_to_snake(col) for col in data.columns]

	print "_"*50
	print "Summary Statistics "+str(variable)+": "
	count = (data[str(variable)].count())
	Number_variable_lines = line_count(dataset)-1
	print "Missing values: ", (Number_variable_lines - count)
	print "Describe "+str(variable)+": ", '\n', (data[str(variable)].describe())
	print "Mode: ", (data[str(variable)].mode())
	#Histogram
	try:
		if count > 1:
			histogram1(str(variable), dataset, 'c', bin1)
			histogram1(str(variable), dataset, 'g', bin2)
			histogram2(str(variable), dataset, 'b', (bin1/float(4)), bin2)
			histogram2(str(variable), dataset, 'r', (bin1/float(5)), bin2)
	except:
		pass




#______________ LOAD and DESCRIBE DATA __________________________#

# Describe Dataset Lines and Variables
dataset_describe(dataset) 

# Load Data to Pandas
data = pd.read_csv(dataset, index_col=0, low_memory=False)
data.columns = [camel_to_snake(col) for col in data.columns]

# Generate Summary Statistics
for col in data.columns:
	summary_statistics(col, 'data/cohort1_all.csv', 5, 10)



#summarize_dataset('data/cs-training.csv')

#bar('serious_dlqin2yrs', 'data/cs-training.csv')
#bar('serious_dlqin2yrs', dataset)


