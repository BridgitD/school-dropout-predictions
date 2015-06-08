'''
Christine Cook
Machine Learning

Notes:
RF sometimes has 0 for precision.recall/f1
LReg accuracy is weird
add k-folds
add timing
'''
from IPython import embed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree, datasets, linear_model
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier  
import time
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB


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

def cleanData(data, cohort):
    if cohort == 1:
        dropList = ['g6_tardyr','g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_gradeexp', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_gradeexp', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_gradeexp', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_gradeexp', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_gradeexp', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_gradeexp', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_gradeexp', 'g12_grade', 'g12_wcode']
        data.drop(dropList, axis=1, inplace=True)
        
    elif cohort == 2:
        dropList = ['g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_grade', 'g12_wcode']
        data.drop(dropList, axis=1, inplace=True)

    #drop id, school id, school name
    data.drop('id', axis=1, inplace=True)
    school_ids = [col for col in data.columns if 'school_id' in col]
    school_names = [col for col in data.columns if 'school_name' in col]
    data.drop(school_ids, axis=1, inplace=True)
    data.drop(school_names, axis=1, inplace=True)

    ##clean birth year/mo
    data.loc[:, 'g11_byrmm']= data.loc[:,'g11_byrmm'].astype(str)
    data.loc[:, 'birth_year'] = data['g11_byrmm'].str[0:4]
    data.loc[:, 'birth_mo'] = data['g11_byrmm'].str[4:6]

    birthday_cols = ['g11_byrmm', 'g12_byrmm', 'g10_byrmm', 'g9_byrmm', 'g8_byrmm', 'g7_byrmm', 'g6_byrmm']
    for col in birthday_cols:
        data.loc[:, col]= data.loc[:,col].astype(str)
        data['birth_year'].fillna(data[col].str[0:4], inplace=True)
        data['birth_mo'].fillna(data[col].str[4:6], inplace=True)

    data.drop(birthday_cols, axis=1, inplace=True)
    
    #clean gender
    data['gender'] = data['g11_gender']
    gender_cols = ['g12_gender', 'g11_gender', 'g10_gender', 'g9_gender', 'g8_gender', 'g7_gender', 'g6_gender']
    for col in gender_cols:
        data['gender'] = data['gender'].fillna(data[col], inplace=True)
    
    data.drop(gender_cols, axis=1, inplace=True)

    #clean retained
    retained_cols = ['g11_retained', 'g12_retained', 'g9_newmcps', 'g10_newmcps', 'g11_newmcps', 'g12_newmcps', 'g9_newus', 'g10_newus', 'g11_newus', 'g12_newus']
    for col in retained_cols:
        data[col] = data[col].notnull()

    #create flag if a given student is missing a year's worth of data
    grade_id = ['g6_pid', 'g7_pid', 'g8_pid', 'g9_pid', 'g10_pid', 'g11_pid', 'g12_pid']
    year = 6
    for grade_col in grade_id:
        miss_col = 'g' + str(year) + '_missing'
        data[miss_col] = data[grade_col].isnull()
        data.drop(grade_col, axis=1, inplace=True)
        year+=1

    return data

def makeDummies(data):
    data = pd.get_dummies(data, dummy_na=True)

    return data

def limitRows(data, pred_grade):
    #get rid of previous dropouts
    for x in range(6, pred_grade-1):
        data = data[data.g6_dropout !=1]
        data = data[data.g7_dropout !=1]
        data = data[data.g8_dropout !=1]
        data = data[data.g9_dropout !=1]
        if pred_grade >= 10:
            data = data[data.g10_dropout !=1]
            if pred_grade >= 11:
                data = data[data.g11_dropout !=1]

    return data

def chooseCols(data, pred_grade):
    #drop 'future' vars
    for x in range(pred_grade, 13):
        dropVars = [col for col in data.columns if str(x) in col]
        dropoutVar = 'g' + str(x) + '_dropout'
        if dropoutVar in dropVars:
            dropVars.remove(dropoutVar)

        data.drop(dropVars, axis=1, inplace=True)

    #drop irrelevent d/o vars
    colList = [col for col in data.columns if 'dropout' in col]
    doVar = 'g' + str(pred_grade) + '_dropout'
    colList.remove(doVar)
    data.drop(colList, axis=1, inplace=True)

    return data

def imputeData(data):
    #change msam to missing is msam_NA==1
    nanList =  ['g6_g6msam_nan', 'g7_g7msam_nan', 'g8_g8msam_nan', 'g9_g8msam_nan']
    msamList = [[ 'g6_g6msam_Advanced', 'g6_g6msam_Basic', 'g6_g6msam_Proficient'], ['g7_g7msam_Advanced', 'g7_g7msam_Basic', 'g7_g7msam_Proficient'], ['g8_g8msam_Advanced', 'g8_g8msam_Basic', 'g8_g8msam_Proficient'],['g9_g8msam_Advanced', 'g9_g8msam_Basic', 'g9_g8msam_Proficient']]
    for x in range(0,len(nanList)):
        nacol = nanList[x]
        colList = msamList[x]
        for col in colList:
            data.loc[data[nacol] == 1, col] = np.nan 


    #pred missing data using any available data
    wordList = ['absrate', 'mapr', 'msam_Advanced', 'msam_Basic', 'msam_Proficient', 'mobility', 'nsusp', 'mpa', 'tardyr', 'psatm', 'psatv', 'retained']
    for word in wordList:
        colList = [col for col in data.columns if word in col]
        rowMean = data[colList].mean(axis=1)
        for col in colList:
            data[col] = data[col].fillna(rowMean)

    return data

def plotROC(name, probs, test_data):
    fpr, tpr, thresholds = roc_curve(test_data['g12_dropout'], probs)
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

def fitClf(clf, x_train, y_train, x_test):
    embed()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test).round()
    #probs = clf.predict_proba(x_test)
    return preds

def getScores(clf_results, name, clf, y_test, preds, x_test):
    precision = precision_score(y_test, preds) 
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    accuracy = clf.score(x_test, y_test)
    clf_results[name] = {}
    clf_results[name]['accuracy'] = accuracy
    clf_results[name]['precision'] = precision
    clf_results[name]['recall'] = recall
    clf_results[name]['f1'] = f1
    print clf_results[name]
    return clf_results


  

def main():
    #read data
    train_data = pd.read_csv('/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort1_all_school.csv', index_col=False)
    test_data = pd.read_csv('/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort2_all_school.csv', index_col=False)

    #prepare data for model
    cohort=1
    for data in [train_data, test_data]:
        #clean data
        data = cleanData(data, cohort)
        #make dummies
        data = makeDummies(data)
        #limit rows to valid
        data = limitRows(data, 12)
        #shrink dataset size
        data = chooseCols(data, 12)
        #impute data 
        data = imputeData(data)
        #drop data if still missing
        data = data[data['g12_dropout'].notnull()]
        #mean-impute the rest
        for col in data.columns.tolist():
            data[col] = data[col].fillna(value=data[col].mean())

        cohort+=1


    # define parameters
    names = ["Nearest Neighbors", "Linear SVM", "Decision Tree", "Random Forest", "AdaBoost", "Linear Regression", "Bagging", "Logistic Regression", "Stochastic Gradient Descent"]
    classifiers = [KNeighborsClassifier(3), LinearSVC(C=0.025), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), AdaBoostClassifier(), linear_model.LinearRegression(), BaggingClassifier(), linear_model.LogisticRegression(), SGDClassifier(loss="hinge", penalty="l2")]

    #start k-fold
    #train_data, test_data = train_test_split(data, test_size=.2)

    # define xs, y
    colList = data.columns.tolist()
    colList.remove('g12_dropout')
    x_train, x_test = train_data.loc[:,colList], test_data.loc[:,colList]
    y_train, y_test = train_data.loc[:,'g12_dropout'], test_data.loc[:,'g12_dropout']

    clf_results = {}

    #loop through classifiers, get predictions, scores
    for name, clf in zip(names, classifiers):
        #fit clf
        preds = fitClf(clf, x_train, y_train, x_test)

        # evaluate classifier, add results to dict
        clf_results = getScores(clf_results, name, clf, y_test, preds, x_test)

 
    print "End"

main()
