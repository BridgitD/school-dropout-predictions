##################################
##								##
##	Machine Learning Pipeline 	##
##	Bridgit Donnelly			##
##								##
##################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import time
import sklearn as sk
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

# Code adapted from https://github.com/yhat/DataGotham2013/

'''
NOTE: Main function for this file only loads and produces summary statistics in 
order to allow the user to make decisions about data imputation and feature generation
based on the user's understanding of the data.
'''

## STEP 1: READ DATA IN CSV FORM

def read_data(filename):
	'''
	Takes a filename and returns a dataframe.
	'''
	original = pd.read_csv(filename, header=0)
	df = original.copy()
	return df

# ---------------------------------------------------------------------

## STEP 2: EXPLORE DATA - generate distributions and data summaries

def summarize(df):
	'''
	Given a dataframe and the original filename, this outputs a CSV with an adjusted filename.
	Return the summary table.
	'''

	# Create Summary Table
	summary = df.describe().T

	# Add Median & Missing Value Count
	summary['median'] = df.median()
	summary['missing_vals'] = df.count().max() - df.describe().T['count'] # Tot. Count - count
	
	output = summary.T
	return output

def print_to_csv(df, filename):
	''' Given a dataframe and a filename, outputs a CSV.'''

	# Check that filename was CSV
	filename_split = str(filename).strip().split('.')

	if filename_split[-1] != 'csv':
		filename = "".join(filename_split[0:len(filename_split)-1]) + ".csv"

	df.to_csv(filename)


### JOSH'S FUNCTIONS HERE ###


def histogram(df, field):
	'''Given a dataframe and a field, this creates histograms.'''

	fn = 'histograms/' + field + '.png'

	#Determine number of bins based on number of different values
	unique_vals = len(df[field].value_counts())
	if unique_vals == 2:
		df.groupby(field).size().plot(kind='bar')
	elif unique_vals < 15:
		bins = unique_vals	
		df[field].hist(xlabelsize=10, ylabelsize=10, bins=unique_vals)
	else:
		df[field].hist()

	pylab.title("Distribution of {0}".format(field))
	pylab.xlabel(field)
	pylab.ylabel("Counts")
	pylab.savefig(fn)
	pylab.close()

# ---------------------------------------------------------------------

## STEP 3: PRE-PROCESS DATA - Fill in misssing values

def replace_with_value(df, variables, values):
	'''
	For some variables, we can infer what the missing value should be.
	This function takes a dataframe and a list of variables that match
	this criteria and replaces null values with the specified value.
	'''
	for i in range(len(variables)):
		variable = variables[i]
		value = values[i]
		df[variable] = df[variable].fillna(value)

def replace_with_mean(df, variables):
	'''
	For some variables, imputing missing values with the variable mean 
	makes the most sense.
	This function takes a dataframe and a list of variables that match 
	this criteria and replaces null values with the variable's mean.
	'''
	# Find means for each field
	for field in variables:
		mean = df[field].mean()

		# Replace Null Values
		for index, row in df.iterrows():
			if pd.isnull(row[field]):
				df.ix[index, field] = mean

def replace_with_other_col(df, variable_missing, variable_fill):
	'''
	Takes a variable and replaces missing values (variable_missing) with
	data from another column (variable_fill).
	'''

	for index, row in df.iterrows():
		if pd.isnull(row[variable_missing]):
			df.ix[index, variable_missing] = row[variable_fill]

##################
## TO ADD LATER ##
def replace_class_conditional(df, variables, field):
	'''
	For some variables, we want to impute the missing values by certain
	class-conditional means.
	This function takes a dataframe and a list a variables that match this 
	criteria, as well as the field to use to determine classes. It finds 
	each unique value of the provided class and imputes mean values based on 
	those classes.
	'''
	pass
##################

def impute(df, variable, cols):
	'''
	For some variables, we cannot infer the missing value and replacing with
	a conditional mean does not make sense.
	This function takes a dataframe and a variables that matches this criteria 
	as well as a list of columns to calibrate with and uses nearest neighbors 
	to impute the null values.
	'''
	# Split data into test and train for cross validation
	is_test = np.random.uniform(0, 1, len(df)) > 0.75
	train = df[is_test==False]
	validate = df[is_test==True]
	
	## Calibrate imputation with training data
	imputer = KNeighborsRegressor(n_neighbors=1)

	# Split data into null and not null for given variable
	train_not_null = train[train[variable].isnull()==False]
	train_null = train[train[variable].isnull()==True]

	# Replace missing values
	imputer.fit(train_not_null[cols], train_not_null[variable])
	new_values = imputer.predict(train_null[cols])
	train_null[variable] = new_values

	# Combine Training Data Back Together
	train = train_not_null.append(train_null)

	# Apply Nearest Neighbors to Validation Data
	new_var_name = variable + 'Imputed'
	validate[new_var_name] = imputer.predict(validate[cols])
	validate[variable] = np.where(validate[variable].isnull(), validate[new_var_name],
								validate[variable])


	# Drop Imputation Column & Combine Test & Validation
	validate.drop(new_var_name, axis=1, inplace=True)
	df = train.append(validate)

	''' FOR FUTURE
	# Apply Imputation to Testing Data
	test_df[new_var_name] = imputer.predict(test_df[cols])
	test_df[variable] = np.where(test_df[variable].isnull(), test_df[new_var_name],
								test_df[variable])
	'''

	return df.sort_index()
	

# ---------------------------------------------------------------------

## STEP 4: GENERATE FEATURES - write a function to discretize a continuous variable
## and a function that can take a categorical variable and create binary variables.

def find_features(df, features, variable):
	'''Uses random forest algorithm to determine the relative importance of each
	potential feature. Takes dataframe, a numpy array of features, and the dependent
	variable. Outputs dataframe, sorting features by importance'''
	clf = RandomForestClassifier()
	clf.fit(df[features], df[variable])
	importances = clf.feature_importances_
	sort_importances = np.argsort(importances)
	rv = pd.DataFrame(data={'variable':features[sort_importances],
							'importance':importances[sort_importances]})
	return rv

def adjust_outliers(x, cap):
	'''Takes series and creates upperbound cap to adjust for outliers'''
	if x > cap:
		return cap
	else:
		return x

def bin_variable(df, variable, num_bins, labels=None):
	'''Discretizes a continuous variable based on specified number of bins'''
	new_label = variable + '_bins'
	df[new_label] = pd.cut(df[variable], bins=num_bins, labels=labels)

def get_dummys(df, cols):
	'''Creates binary variable from specified column(s)'''
	# Loop through each variable
	for variable in cols:
		dummy_data = pd.get_dummies(df[variable], prefix=variable)
		
		# Add new data to dataframe
		df = pd.concat([df, dummy_data], axis=1)


# ---------------------------------------------------------------------

## STEP 5: BUILD AND EVALUATE CLASSIFIERS - Create comparison table of the performace
## of each classifier on each evaluation metric

def build_classifiers(X, y):
	'''
	Takes a dataframe of features (X) and a dataframe of the variable to predict (y). 
	Returns a new dataframe comparing each classifier's performace on 
	the given evaluation metrics.
	'''
	rv = pd.DataFrame()

	# Classifiers to test
	classifiers = [('logistic_regression', LogisticRegression()),
					('k_nearest_neighbors', KNeighborsClassifier()),
					('decision_tree', DecisionTreeClassifier()),
					('SVM', LinearSVC()),
					('random_forest', RandomForestClassifier()),
					('boosting', GradientBoostingClassifier()),
					('bagging', BaggingClassifier())]

	index = 0
	for name, clf in classifiers:
		print name

		# Construct K-folds
		kf = KFold(len(y), n_folds=5, shuffle=True)
		y_pred = y.copy()

		rv.loc[index,'classifier'] = name
		start_time = time.time()
		
		# Iterate through folds
		for train_index, test_index in kf:
			X_train, X_test = X[train_index], X[test_index]
			y_train = y[train_index]

			#Initialize classifier
			model = clf
			model.fit(X_train, y_train)
			y_pred[test_index] = model.predict(X_test)
		
		end_time = time.time()
		#print pd.value_counts(predictions)

		# Calculate running time
		rv.loc[index, 'time'] = end_time - start_time

		evaluate_classifier(rv, index, y, y_pred)
		index += 1

	return rv

		
def evaluate_classifier(df, index, y_real, y_predict):
	'''
	For an index of a given classifier, evaluate it by various metrics
	'''
	# Metrics to evaluate
	metrics = [('accuracy', accuracy_score(y_real, y_predict)),
				('precision', precision_score(y_real, y_predict)),
				('recall', recall_score(y_real, y_predict)),
				('f1', f1_score(y_real, y_predict)),
				('area_under_curve', roc_auc_score(y_real, y_predict))]

	for name, m in metrics:
		df.loc[index, name] = m


# -------------------------------------------------------------------------
if __name__ == '__main__':
	
	if len(sys.argv) <= 1:
		sys.exit("Must include a filename.")

	else:
		dataset = sys.argv[1]
		
		## Load data 
		df = read_data(dataset)
		variables = list(df.columns.values)

		## Run initial summary statistics & graph distributions
		summary = summarize(df)
		#print_to_csv(summary, 'summary_stats.csv')
		
		for v in variables:
			histogram(df, v)

		


