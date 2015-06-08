'''
Christine Cook
Machine Learning
'''
from IPython import embed
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import re
import random
from sklearn.cross_validation import KFold
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, f1_score
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import time


def getSumStats(data):
    desc = data.iloc[:,1:].describe().T
    desc.drop([desc.columns[4], desc.columns[6]], axis=1, inplace=True)
    mode = data.iloc[:,1:].mode()
    desc = pd.concat([desc.T, mode])
    desc.rename({0:'mode', '50%':'median'}, inplace=True)
    desc.to_csv("data_sumstats.csv")

def cleanData(data, cohort):
    if cohort == 1:
        dropList = ['g6_tardyr','g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_gradeexp', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_gradeexp', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_gradeexp', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_gradeexp', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_gradeexp', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_gradeexp', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_gradeexp', 'g12_grade', 'g12_wcode']
        data.drop(dropList, axis=1, inplace=True)
        
    elif cohort == 2:
        dropList = ['g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_grade', 'g12_wcode']
        data.drop(dropList, axis=1, inplace=True)

    ##clean birth year/mo
    data.loc[:, 'g11_byrmm']= data.loc[:,'g11_byrmm'].astype(str)
    data.loc[:, 'birth_year'] = data['g11_byrmm'].str[0:4]
    data.loc[:, 'birth_mo'] = data['g11_byrmm'].str[4:6]

    birthday_cols = ['g11_byrmm', 'g12_byrmm', 'g10_byrmm', 'g9_byrmm', 'g8_byrmm', 'g7_byrmm', 'g6_byrmm']
    for col in birthday_cols:
        data.loc[:, col]= data.loc[:,col].astype(str)
        data['birth_year'].fillna(data[col].str[0:4], inplace=True)
        data['birth_mo'].fillna(data[col].str[4:6], inplace=True)

    data.drop('id', axis=1, inplace=True)

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
    for g in grade_id:
        col_name = 'g' + str(year) + '_missing'
        data[col_name] = data[g].isnull()
        data.drop(g, axis=1, inplace=True)
        year+=1

    return data

def makeDummies(data):
    school_ids = [col for col in data.columns if 'school_id' in col]
    data[school_ids] = data.loc[:,school_ids].astype(str, copy=False)

    data = pd.get_dummies(data, dummy_na=True)

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
            data[col].fillna(rowMean, inplace=True)

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

def makeFinite(data, pred_grade):
    #keep finite
    colList = [col for col in data.columns if 'dropout' in col]
    doVar = 'g' + str(pred_grade) + '_dropout'
    colList.remove(doVar)
    data.drop(colList, axis=1, inplace=True)
    data = data.dropna(axis=0)
    return data

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

def imputeMean(data):
    data.fillna(value=data.mean(), inplace=True)
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

def clf_cv_loop(classifier, x_data, y_data):
    poss_class_y_pred = []
    poss_times = []
    for k in classifier['kwords_list']:
        t0 = time.time()
        poss_class_y_pred.append(run_cv(x_data, y_data, classifier['class'], k))
        t1 = time.time()
        total = t1-t0
        poss_times.append(total)
    return poss_class_y_pred[0][1], poss_times, poss_class_y_pred[0][0]

def run_cv(x, y, clf_class, *args, **kwargs):
    embed()
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    y_pred_proba = y.copy()
    # Iterate through folds
    for train_index, test_index in kf:
        x_train = x.ix[train_index]
        x_test  = x.ix[test_index]
        y_train = y.ix[train_index]
        x_train = Imputer(strategy = 'median').fit_transform(x_train)
        x_test = Imputer(strategy = 'median').fit_transform(x_test)
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(x_train,y_train)
        y_pred[test_index] = clf.predict(x_test)
        y_pred_proba[test_index] = clf.predict_proba(x_test)
    return y_pred, y_pred_proba

def eval_clfs(y_pred, y_data, evals, classifier, classifier_name, poss_times, y_pred_proba):
    #embed()
    f = open('./output/'+classifier_name+'_evals_table.csv', 'w')
    f.write('parameters\ttime\t')
    for k, l in evals.iteritems():
        f.write(k+'\t')
    f.write('\n')
    for k in range(len(y_pred)):
        f.write(str(classifier['kwords_list'][k])+'\t')
        f.write(str(posslen_times[k])+'\t')
        for l, m in evals.iteritems():
            if l == 'precision_recall_curve':
                eval_temp = m(y_data, y_pred_proba)
                f.write(str(eval_temp)+'\t')
            else:
                eval_temp = m(y_data, y_pred[k])
                f.write(str(eval_temp)+'\t')
        f.write('\n')
    f.close()


def main():
    #read data
    data = pd.read_csv('/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort1_all.csv', index_col=False)

    #clean data
    data = cleanData(data, 1)

    #make dummies
    data = makeDummies(data)

    #limit rows to valid
    data = limitRows(data, 12)

    #shrink dataset size
    data = chooseCols(data, 12)

    #impute data 
    data = imputeData(data)

    #make data finite
    #data = makeFinite(data, 12) 
    data.dropna(axis=0, inplace=True)

    #define features
    features = data.columns.tolist()
    features.remove('g12_dropout')
    
    #define classifiers
    classifiers =   {
                    #'LogisticRegression': {'class': LogisticRegression}, 
                    #'KNeighborsClassifier': {'class': KNeighborsClassifier}, 
                    'DecisionTreeClassifier': {'class': DecisionTreeClassifier}} 
                    #'LinearSVC': {'class': LinearSVC}, 
                    #'RandomForestClassifier': {'class': RandomForestClassifier}, 
                    #'AdaBoostClassifier': {'class': AdaBoostClassifier}, 
                    #'BaggingClassifier': {'class': BaggingClassifier}}

    #define eval metrics
    evals = {'accuracy_score': accuracy_score, 
            'precision_score': precision_score, 
            'recall_score': recall_score, 
            'f1_score': f1_score, 
            'roc_auc_score': roc_auc_score, 
            'precision_recall_curve': precision_recall_curve}
    
    #Creating lists to loop over for parameters
    #for i in range(10):
       #temp = classifiers['KNeighborsClassifier'].get('kwords_list', [])
       #temp.append({'n_neighbors': i})
       #classifiers['KNeighborsClassifier']['kwords_list'] = temp
    for i in range(1,6,1):
        temp = classifiers['DecisionTreeClassifier'].get('kwords_list', [])
        temp.append({'max_depth': i})
        classifiers['DecisionTreeClassifier']['kwords_list'] = temp

    '''
    for i in range(2,22,2):
        temp = classifiers['RandomForestClassifier'].get('kwords_list', [])
        temp.append({'n_estimators': i})
        classifiers['RandomForestClassifier']['kwords_list'] = temp
    for i in range(50, 110, 10):
        temp = classifiers['AdaBoostClassifier'].get('kwords_list', [])
        temp.append({'n_estimators': i})
        classifiers['AdaBoostClassifier']['kwords_list'] = temp
    for i in range(6, 16, 2):
        temp = classifiers['BaggingClassifier'].get('kwords_list', [])
        temp.append({'n_estimators': i})
        classifiers['BaggingClassifier']['kwords_list'] = temp

    #classifiers['LogisticRegression']['kwords_list'] = [{'C': 1.0}]
    #classifiers['LSVC']['kwords_list'] = [{'C': 1.0}]
    '''
    #define x, y
    x_data = data[features]
    y_data = data['g12_dropout']

    #run clf
    for i, j in classifiers.iteritems():
        y_pred, poss_times, y_pred_proba = clf_cv_loop(j, x_data, y_data)
        eval_clfs(y_pred, y_data, evals, j, i, poss_times, y_pred_proba)






    print "End"

main()
