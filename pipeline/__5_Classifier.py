"""
Joshua Mausolf - CAPP 30254 Assignment pa3.

In this python module I develop numerous classifier functions, including
logit, SVMs, KNN, Decision Tree, Random Forests, Random Forests with Bagging/Boosting, and
Gradient Boosting among others.

I have hashed these functions, which can be unhashed to test.
I evaluate these classifiers in __6_Evaluate.py. 
"""

import sys, os, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score


# Import Classifiers 
from sklearn import svm, preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier


#Boosting and Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier

#Time
import time
from threading import Timer



#______________ LOAD and DESCRIBE DATA __________________________#

#Choose Dataset
dataset = 'data/cs-training#3B.csv' #Post-impute data


def camel_to_snake(column_name):
    """
    Converts a string that is camelCase into snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


#______________ CLASSIFIER FUNCTIONS ____________________________#


# ___ LOGIT ___ #

def logistic_regression_classifier(dataset, DV, train='no'):
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	model = LogisticRegression()

	if train=='yes':
		model.fit(X_train, y_train)
		return model, X, y, X_train, X_test, y_train, y_test
	else:
		model = model.fit(X, y)
		return model, X, y



def logistic_regression_table(dataset, DV, table='no', CSV='no'):
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	logit = sm.Logit(y, X)
	model = logit.fit()

	#Prediction
	prediction = 'prediction_'+str(DV)
	data[prediction] = model.predict(X)

	#Redidual
	data['residual_'+str(DV)] = data[str(DV)] - data[prediction]
	#print data.head()

	if table == 'table':
		model_output = model.summary()
		odds_ratio = '\n'+"Odd's Ratios:"+'\n'+str(np.exp(model.params))+'\n'
		f = open("output/logistic_regression_table.txt", 'w')
		f.write(str(model_output))
		f.write(str(odds_ratio))
		f.close

		#Write CSV Option
		if CSV == 'csv':
			data.to_csv('data/logit_data_prediction_residuals.csv', encoding='utf-8')
		else:
			pass


	else:
		print model.summary()
		print "Odd's Ratios:", '\n', np.exp(model.params), '\n'

	
		#Write CSV Option
		if CSV == 'csv':
			data.to_csv('data/logit_data_prediction_residuals.csv', encoding='utf-8')
		else:
			pass


# Unhash to test:
#logistic_regression_table(dataset, 'serious_dlqin2yrs', 'table')
#logistic_regression_table(dataset, 'serious_dlqin2yrs', 'table', 'csv')


def logit_clf(dataset, DV, train):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

	model = LogisticRegression()
	logistic_regression_table(dataset, DV, 'table')
	
	if train=='yes':
		model1 = model.fit(X_train, y_train)
		print "Classifier: Logistic Regression"
		print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
		end = time.time()
		print "Runtime, base model: %.3f" % (end-start), "seconds."
		return model1
	elif train=='no':
		model2 = model.fit(X, y)
		print "Classifier: Logistic Regression"
		print pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))
		end = time.time()
		print "Runtime, base model: %.3f" % (end-start), "seconds."
		return model2


# Unhash to test:
#print logit_clf('data/cs-training#3B.csv', 'serious_dlqin2yrs')


# ___ LINEAR SVMs ___ #

def Build_Data_Set(dataset, DV, lower_limit=0, upper_limit=''):
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	X = preprocessing.scale(X)

	return X[lower_limit:upper_limit],y[lower_limit:upper_limit]

def lin_svc(dataset, DV, lower_limit=0, upper_limit=''):
	"""Runs a linear for a given DV, using all remaining
	variables as features.
	Prohibitive run time for N > 20,000. Cross-validation even worse.
	Consider anova_svm instead.""" 



	start = time.time()

	X, y = Build_Data_Set(dataset, DV, lower_limit, upper_limit)

	clf = SVC(kernel="linear", C= 1.0)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: Linear SVC"
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 


# Unhash to test:
#lin_svc('data/cs-training#3B.csv', 'serious_dlqin2yrs', 0, 150000)


def anova_svm(dataset, DV, k, lower_limit=0, upper_limit=''):
	start = time.time()

	X, y = Build_Data_Set(dataset, DV, lower_limit, upper_limit)

	# ANOVA SVM-C
	# 1) anova filter, take 3 best ranked features
	anova_filter = SelectKBest(f_regression, k=3)
	# 2) svm
	clf = SVC(kernel='linear', probability=True, C=0.5, cache_size=10000)

	anova_svm = make_pipeline(anova_filter, clf)
	model = anova_svm.fit(X, y)

	end = time.time()
	print "Classifier: Anova Linear SVM, "+str(k)+"-best features"
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 


# Unhash to test:
#print anova_svm('data/cs-training#3B.csv', 'serious_dlqin2yrs', 3, 0, 20000)


# ___ DECISION TREE ___ #

def d_tree(dataset, DV, max_dep):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]


	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]


	clf = tree.DecisionTreeClassifier(max_depth=max_dep)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: Decision Tree,", "Depth = ", max_dep
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 

# Unhash to test:
#d_tree('data/cs-training#3B.csv', 'serious_dlqin2yrs', 5)


# ___ RANDOM FOREST ___ #

def random_forest(dataset, DV, max_dep):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	clf = RandomForestClassifier(n_jobs=2, max_depth=max_dep)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: Random Forest,", "Depth = ", max_dep
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 

# Unhash to test:
#random_forest('data/cs-training#3B.csv', 'serious_dlqin2yrs', 5)


# ___ BAGGING ___ #

def random_forest_bagging(dataset, DV, max_dep):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	clf = BaggingClassifier(RandomForestClassifier(n_jobs=2, max_depth=max_dep), max_samples=0.5, max_features=0.5)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: Random Forest with Bagging,", "Depth = ", max_dep
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 


# Unhash to test:
#random_forest_bagging('data/cs-training#3B.csv', 'serious_dlqin2yrs', 5)


# ___ BOOSTING ___ #

def random_forest_boosting(dataset, DV, max_dep):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	clf = AdaBoostClassifier(RandomForestClassifier(n_jobs=2, max_depth=max_dep), n_estimators=20)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: Random Forest with Boosting,", "Depth = ", max_dep
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 

# Unhash to test:
#random_forest_boosting('data/cs-training#3B.csv', 'serious_dlqin2yrs', 2)

def gradient_boosting(dataset, DV, max_dep):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=max_dep, random_state=0)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: Gradient Boosting,", "Depth = ", max_dep
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 

# Unhash to test:
#gradient_boosting('data/cs-training#3B.csv', 'serious_dlqin2yrs', 3)


# ___ KNN ___ #

def KNN(dataset, DV, neighbors, method='auto'):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]


	#DV
	y = data[str(DV)]
	X = data[data.columns - [str(DV)]]

	clf = KNeighborsClassifier(n_neighbors=neighbors, algorithm=method, leaf_size=5)
	model = clf.fit(X, y)

	end = time.time()
	print "Classifier: KNN,", "neighbors = ", neighbors, ", algorithm = ", method
	print "Runtime, base model: %.3f" % (end-start), "seconds."
	return model 

# Unhash to test:
#KNN('data/cs-training#3B.csv', 'serious_dlqin2yrs', 3, 'kd_tree')










