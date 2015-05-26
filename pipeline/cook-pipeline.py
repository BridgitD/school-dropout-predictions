'''
Christine Cook
Machine Learning
PA 3
'''

import pandas as pd 
import numpy as np
import pylab as pl 
import csv, time
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, datasets, linear_model
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


def getSumStats(data):
    desc = data.iloc[:,1:].describe().T
    desc.drop([desc.columns[4], desc.columns[6]], axis=1, inplace=True)
    mode = data.iloc[:,1:].mode()
    desc = pd.concat([desc.T, mode])
    desc.rename({0:'mode', '50%':'median'}, inplace=True)
    desc.to_csv("data_sumstats.csv")

def makeChartDiscrete(data, col, title):
    data_copy = data
    data_copy = data_copy.dropna()
    data_max = data_copy.iloc[:,col].max()
    step = (data_max/50)
    if step < 1:
        bins=list(range(0, int(data_max), 1))
    else:
        bins=list(range(0, int(data_max), step))
    pl.figure()
    pl.title(title)
    pl.xlabel(title)
    pl.ylabel('Frequency')
    bins = pl.hist(data_copy.iloc[:,col], bins)
    pl.savefig(title)

def makeChartContinuous(data, col, title):
    y_vals = data.iloc[:,col]
    data_id = data.iloc[:,0]
    pl.figure()
    pl.title(title)
    pl.xlabel(title)
    pl.ylabel('Frequency')
    pl.scatter(y_vals,data_id)
    pl.savefig(title)

def imputeConditionalMean(data, col):
    full_data = pd.DataFrame()
    yes = data[data[col] == 1].fillna(data[data[col] == 1].mean())
    no = data[data[col] == 0].fillna(data[data[col] == 0].mean())
    full_data = pd.concat([yes, no])
    return full_data

def imputeMean(data):
    data.fillna(value=data.mean(), inplace=True)
    return data

def discretize(data, col, num_bins):
    data[col] = pd.cut(data[col], bins=num_bins, labels=False)
    return data

def plotROC(name, probs, test_data):
    fpr, tpr, thresholds = roc_curve(test_data['SeriousDlqin2yrs'], probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name) 
    pl.legend(loc="lower right")
    pl.savefig(name)


def evaluateClassifier(name, y_true, y_pred, probs, test_data):
    # precision, recall, F1 scores, accuracy
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # ROC curve, AUC on fig
    plot_roc("Perfect Classifier", test_data['SeriousDlqin2yrs'], test_data)
    plot_roc("Guessing", np.random.uniform(0, 1, len(test_data['SeriousDlqin2yrs'])), test_data)
    plotROC(name, probs, test_data)
    return precision, recall, f1



def main():
    #read data
    data = pd.read_csv('cs-training.csv', index_col=False)

    # define parameters
    names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "AdaBoost", "Linear Regression", "Bagging"]
    classifiers = [KNeighborsClassifier(3), LinearSVC(C=0.025), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), AdaBoostClassifier(), linear_model.LinearRegression(), BaggingClassifier()]

    # split data
    for x in range(0,5):
        print "\nSplit: " + str(x)

        train_data, test_data = train_test_split(data, test_size=.2)

        # impute missing data using mean
        train_data = imputeMean(train_data)
        test_data = imputeMean(test_data)

        # define Xs, Y
        X_train = train_data.iloc[:,2:]
        y_train = train_data.iloc[:,1]
        X_test = test_data.iloc[:,2:]
        y_test = test_data.iloc[:,1]

        clf_results = {}

        # loop through classifiers, get predictions, scores
        for name, clf in zip(names, classifiers):

            #time training
            start_time = time.clock()
            clf.fit(X_train, y_train)
            end_time = time.clock()
            training_time = (end_time - start_time)

            #time testing
            start_time = time.clock()
            if (name=="Linear Regression") | (name=="Linear SVM"):
                probs = clf.predict(X_test)
                preds = probs.round()
            else:
                preds = clf.predict(X_test)
                probs = clf.predict_proba(X_test)[::,1]
            end_time = time.clock()
            testing_time = (end_time - start_time)

            # evaluate classifier
            precision, recall, f1 = evaluateClassifier(name, y_test, preds, probs, test_data)
            accuracy = clf.score(X_test, y_test)

            # add results to dict
            clf_results[name] = {}
            clf_results[name]['accuracy'] = accuracy
            clf_results[name]['precision'] = precision
            clf_results[name]['recall'] = recall
            clf_results[name]['f1'] = f1
            clf_results[name]['testing time'] = testing_time
            clf_results[name]['training time'] = training_time


        print clf_results

        with open('trial_' + str(x) + '.csv', 'wb') as f:
            w = csv.DictWriter(f, clf_results.keys())
            w.writeheader()
            w.writerow(clf_results)
    
    print "End"
