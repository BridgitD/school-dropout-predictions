"""
Joshua Mausolf - CAPP 30254 Assignment pa3.

In this python module, I develop several general functions, chiefly:

1. I develop a function that can discretize a continuous variable and return a new data frame.
2. I develop a function that can discretize a list of continous variables and write a new data-file.
3. I define a function that can take a categorical variable and create binary variables from it.

Because there did not appear to be any natural categorical variables in the dataset,
the binary function can take a binary and make two reverse coded dummies. 
For illustration of the categorical functionality, I transformed a continuous variable into 
a categorical variable and then made a series of binary dummy variables. 

I have left several of these functions un-hashed so that the module may be run from the terminal
to produce results. Other examples are provided, but hashed.
"""

import sys, os
import csv
import pandas as pd 
import re
import numpy as np 


def camel_to_snake(column_name):
    """
    Converts a string that is camelCase into snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()



def discretize_quartiles(variable, CSV_IN_file_name):
	VAR = str(variable)
	VAR_discrete = VAR+'_discrete'
	CSV_IN = CSV_IN_file_name+'.csv'
	CSV_OUT = CSV_IN_file_name+'_discrete_'+variable+'.csv'

  	
	data = pd.read_csv(CSV_IN, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]
	data.rename(columns={VAR: VAR_discrete}, inplace=True)
	quantiles = pd.qcut(data[VAR_discrete], 4, labels=['1', '2', '3', '4'])
	data.rename(columns={VAR_discrete: VAR}, inplace=True)
	cols_to_keep = data.columns
	data_new = data[cols_to_keep].join(quantiles)

	#Write Out Data Frame to CSV File
	data_new.to_csv(CSV_OUT, encoding='utf-8')


#Unhash to run
#discretize_quartiles('monthly_income', 'data/cs-training#3B')



def make_discrete_df(variable, df, option=4):
	"""This function augments the discretize function. It takes a single variable, data frame, and
	option and returns a new data frame with both the original variable and the new discrete variable.

		The options include the following: 
		2 	= Median
		4 	= Quartiles
		5 	= Quintiles
		10 	= Deciles
		99	= Distribution with segment splits 0, 0.01, 0.05, 0.5, 0.95, 0.99, and 1

	"""

	OPT = str(option)
	VAR = str(variable)
	VAR_discrete = VAR+'_discrete'+OPT
	#CSV_IN = CSV_IN_file_name+'.csv'

  	
	data = df
	data.columns = [camel_to_snake(col) for col in data.columns]
	data.rename(columns={VAR: VAR_discrete}, inplace=True)

	#Median
	if option == 2:
		median = pd.qcut(data[VAR_discrete], 2, labels=['0', '1'])
		data.rename(columns={VAR_discrete: VAR}, inplace=True)
		cols_to_keep = data.columns
		data_new = data[cols_to_keep].join(median)
		return data_new

	#Quartiles
	elif option == 4:
		quantiles = pd.qcut(data[VAR_discrete], 4, labels=['1', '2', '3', '4'])
		data.rename(columns={VAR_discrete: VAR}, inplace=True)
		cols_to_keep = data.columns
		data_new = data[cols_to_keep].join(quantiles)
		return data_new

	#Quintiles
	elif option ==5:
		quintiles = pd.qcut(data[VAR_discrete], 5, labels=['1', '2', '3', '4', '5'])
		data.rename(columns={VAR_discrete: VAR}, inplace=True)
		cols_to_keep = data.columns
		data_new = data[cols_to_keep].join(quintiles)
		return data_new

	#Deciles
	elif option ==10:
		quantiles = pd.qcut(data[VAR_discrete], 10, labels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
		data.rename(columns={VAR_discrete: VAR}, inplace=True)
		cols_to_keep = data.columns
		data_new = data[cols_to_keep].join(quantiles)
		return data_new

	#Distribution
	elif option ==99:
		quantiles = pd.qcut(data[VAR_discrete], [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1], labels=['0.01', '0.05', '0.5', '0.95', '0.99', '1'])
		data.rename(columns={VAR_discrete: VAR}, inplace=True)
		cols_to_keep = data.columns
		data_new = data[cols_to_keep].join(quantiles)
		return data_new

	else:
		print "Please input a valid option."
		return


def discretize(CSV_IN_file_name, option, args):
	"""This function takes a list of one or more continous variables and makes respective categorical variables. It maintains the original variable.

		The variable name and CSV file must be given as an exact string. The CSV filename omits the extension '.csv.'

		An option must also be specified. 
		2 	= Median
		4 	= Quartiles
		5 	= Quintiles
		10 	= Deciles
		99	= Distribution with segment splits 0, 0.01, 0.05, 0.5, 0.95, 0.99, and 1

	"""

	OPT = str(option)
	CSV_IN = CSV_IN_file_name+'.csv'
	CSV_OUT = CSV_IN_file_name+'_discrete_'+OPT+'.csv'


	data = pd.read_csv(CSV_IN, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#Unwrap arguments
	variable_list = args

	#Loop over variable arguments
	for var in variable_list:
		df = data
		data = make_discrete_df(var, df, option)


	#Write Out Data Frame to CSV File
	data.to_csv(CSV_OUT, encoding='utf-8')



#Unhash to test
discretize('data/cs-training#3B', 4, ['monthly_income', 'age', 'debt_ratio', 'number_of_open_credit_lines_and_loans'])




def binary(variable, CSV_IN_file_name):
	"""	This function takes a variable and makes binary dummy variables for each variable option. It is recommended for categorical or binary variables.Although it will function on a continuous variable, it is recommended to run the <discretize> function first."""

	CSV_IN = CSV_IN_file_name+'.csv'
	CSV_OUT = CSV_IN_file_name+'_dummy_'+variable+'.csv'
	#Read Data
	data = pd.read_csv(CSV_IN, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]
	#Create Binary Dummy Variable
	dummy = pd.get_dummies(data[str(variable)], prefix=str(variable))
	cols_to_keep = data.columns - [str(variable)]
	data_new = data[cols_to_keep].join(dummy)
	#Write Out Data Frame to CSV File
	data_new.to_csv(CSV_OUT, encoding='utf-8')


#Unhash to test
#binary('serious_dlqin2yrs', 'data/cs-training#3B')
#binary('number_of_dependents', 'data/cs-training#3B')
#binary('monthly_income_discrete4', 'data/cs-training#3B_discrete_monthly_income4')


