"""
Joshua Mausolf - CAPP 30254 Assignment pa3.

In this python module I develop the code to evaluate multiple 
machine learning classifiers. I show the results in the
summary report.
"""

import sys, os, re
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt


# IMPORT CLASSIFIERS
from __5_Classifier import *

# SKLEARN EVALUATION
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


# TIME
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



#______________ PRECISION RECALL CURVE ____________________________#

def plot_precision_recall_curve(y_test, y_score, model):
	# Compute Precision-Recall and plot curve
	precision, recall, thresholds = precision_recall_curve(y_test, y_score[:, 1])
	area = auc(recall, precision)
	print "Area Under Curve: %0.2f" % area

	pl.clf()
	pl.plot(recall, precision, label='Precision-Recall curve')
	pl.xlabel('Recall')
	pl.ylabel('Precision')
	pl.ylim([0.0, 1.05])
	pl.xlim([0.0, 1.0])
	pl.title('Precision-Recall example: AUC=%0.2f' % area)
	pl.legend(loc="lower left")

	plt.draw()
	plt.savefig("output/evaluation/"+model+"_precision-recall-curve"+".jpg")
	plt.clf()



#______________ EVALUATION FUNCTIONS ____________________________#



def evalualte_base_split(dataset, DV, model):
	start = time.time()
	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	if model == 'logit':
		#DV
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = logit_clf(dataset, DV, 'yes')


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
	predicted = clf.predict(X_test)
	y_score = clf.predict_proba(X_test)

	
	# generate evaluation metrics
	print "Model score, accuracy : %.3f" % (metrics.accuracy_score(y_test, predicted))
	print "Model score, roc_auc: %.3f" % (metrics.roc_auc_score(y_test, y_score[:, 1]))
	print "Model score, f1: %.3f" % metrics.f1_score(y_test, predicted)
	print "Model score, average-precision: %.3f" % (metrics.average_precision_score(y_test, predicted))
	print "Model score, precision: %.3f" % (metrics.precision_score(y_test, predicted))
	print "Model score, recall: %.3f" % (metrics.recall_score(y_test, predicted))

	end = time.time()
	print "Runtime, K-folds evaluation of base model: %.3f" % (end-start), "seconds."

#evalualte_base_split('data/cs-training#3B.csv', 'serious_dlqin2yrs', 'logit')



def evalualte_K_fold(dataset, DV, model, sets):
	start = time.time()

	# Load Data to Pandas
	data = pd.read_csv(dataset, index_col=0)
	data.columns = [camel_to_snake(col) for col in data.columns]

	#Metrics
	metrics = ['accuracy', 'average_precision', 'precision', 'recall', 'f1', 'roc_auc', 'log_loss', 'mean_squared_error', 'r2']
	
	# ________ Determine which classifier ________ #
	if model == 'logit':
		#DV
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = logit_clf(dataset, DV, 'no')


	elif model == 'lin_svc':
		X, y = Build_Data_Set(dataset, DV, 0, 150000)
		clf = lin_svc(dataset, DV, 0, 150000)


	elif model == 'anova_svm':
		X, y = Build_Data_Set(dataset, DV, 0, 150000)
		clf = anova_svm(dataset, DV, 3, 0, 150000)

	elif model == 'd_tree':
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = d_tree(dataset, DV, 4)

	elif model == 'random_forest':
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = random_forest(dataset, DV, 5)
		#clf = random_forest(dataset, DV, 10)

	elif model == 'random_forest_bagging':
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = random_forest_bagging(dataset, DV, 5)

	elif model == 'random_forest_boosting':
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = random_forest_boosting(dataset, DV, 5)

	elif model == 'gradient_boosting':
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		clf = gradient_boosting(dataset, DV, 3)


	elif model == 'KNN':
		y = data[str(DV)]
		X = data[data.columns - [str(DV)]]
		#clf = KNN(dataset, DV, 3)
		clf = KNN(dataset, DV, 3)

	# ________ Evaluate ________ #
	
	#Base, no split
	if sets==1:
		print "Model score: ", clf.score(X, y)
		print "Baseline = Mean of DV: ", y.mean()
		print "Prediction - null: ", clf.score(X, y) - (1-y.mean())


	# K-fold Cross Validation
	if sets >1:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
		
		try:
			y_score = clf.fit(X_train, y_train).predict_proba(X_test)
			plot_precision_recall_curve(y_test, y_score, model)
		except:
			pass
		print "Cross-validation of "+model+" classifier: "+str(sets)+"-folds"
		print "Baseline = 1-Mean of DV: %.3f" % (1-y.mean())
		for metric in metrics:
			scores = cross_val_score(clf, X, y, scoring=str(metric), cv=sets)
			print "Model mean score, "+metric+" : %.3f" % (scores.mean())
		end = time.time()
		print "Runtime, K-folds evaluation of base model: %.3f" % (end-start), "seconds."



#______________ GENERATE EVALUATIONS ____________________________#


# Unhash to test
classifiers = ['logit', 'd_tree', 'random_forest', 'random_forest_bagging', 'random_forest_boosting', 'gradient_boosting', 'KNN', 'anova_svm']
#classifiers = ['KNN']


# Five-Fold Cross Validation
def EVAL():

	for clf in classifiers:
		print "__"*50, "\n"
		evalualte_K_fold('data/cs-training#3B.csv', 'serious_dlqin2yrs', clf, 5)
		print "__"*50, "\n"

EVAL()