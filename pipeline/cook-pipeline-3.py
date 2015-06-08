'''
Christine Cook
Machine Learning

Notes:
display tree


feature generation
switch to cohort 2 testing
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
from sklearn.tree import DecisionTreeClassifier  
import time
from sklearn import tree
from sklearn.linear_model import SGDClassifier


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
        data['gender'] = data['gender'].fillna(data[col])
    
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
    data = data.convert_objects(convert_numeric=True)
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
        grade = 'g' + str(x)
        dropVars = [col for col in data.columns if grade in col]
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

def imputeConditionalMean(data, col):
    
    full_data = pd.DataFrame()
    yes = data[data[col] == 1].fillna(data[data[col] == 1].mean())
    no = data[data[col] == 0].fillna(data[data[col] == 0].mean())
    full_data = pd.concat([yes, no])

    return full_data

def fitClf(clf, x_train, y_train, x_test):
    train_t0 = time.time()
    clf.fit(x_train, y_train)
    train_t1 = time.time()

    test_t0 = time.time()
    preds = clf.predict(x_test).round()
    test_t1 = time.time()
    probs = clf.predict_proba(x_test)
    return preds, probs, (train_t1-train_t0), (test_t1-test_t0)

def getScores(clf_results, x, name, clf, y_test, preds, x_test, train_time, test_time):
    print name
    precision = precision_score(y_test, preds) 
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    accuracy = clf.score(x_test, y_test)
    clf_results[x][name] = {}
    clf_results[x][name]['accuracy'] = accuracy
    clf_results[x][name]['precision'] = precision
    clf_results[x][name]['recall'] = recall
    clf_results[x][name]['f1'] = f1
    clf_results[x][name]['train_time'] = train_time
    clf_results[x][name]['test_time'] = test_time
    print clf_results[x][name]
    return clf_results[x][name]

def findMisClf(df, X, y, y_pred, name):
    '''
    Takes a dataframe (df), column names of predictors (X) and a dependent
    variable (y). Loops over generic classifiers to find predictions. Creates
    a decision tree using prediction misclassification as the dependent variable.
    '''

    var_name = name + '_predict'
    try:
        df[var_name] = y_pred
    except:
        import pdb
        pdb.set_trace()
    correct = name + '_correct'
    
    # Determine "correctness" based on 0.5 threshold
    df[correct] = (df[var_name] > 0.5).astype(int)

    # Determine which observations are being misclassified
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(df[X.columns], df[correct])
    feature_names = df.columns
    left, right = tree.tree_.children_left, tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value
    tree.export_graphviz(tree, out_file='tree.dot', feature_names = X.columns) 

    def recurse(left, right, threshold, features, node):
            if (threshold[node] != -2):
                    print "if ( " + features[node] + " <= " + str(threshold[node]) + " ) {"
                    if left[node] != -1:
                            recurse (left, right, threshold, features,left[node])
                    print "      } else {"
                    if right[node] != -1:
                            recurse (left, right, threshold, features,right[node])
                    print "}"
            else:
                    print "return " + str(value[node])

    #recurse(left, right, threshold, features, 0)


def main():
    #define constants
    pred_grade = 12
    DV = 'g' + str(pred_grade) + '_dropout'

    #read data
    data = pd.read_csv('/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort1_all_school.csv', index_col=False)
    #test_data = pd.read_csv('/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort2_all_school.csv', index_col=False)
    #clean data
    data = cleanData(data, 1)
    #make dummies
    data = makeDummies(data)
    #limit rows to valid
    data = limitRows(data, pred_grade)
    #shrink dataset size
    data = chooseCols(data, pred_grade)
    #impute data 
    data = imputeData(data)
    #drop data if still missing
    data = data[data[DV].notnull()]


    # define parameters
    names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "AdaBoost", "Bagging", "Logistic Regression", "Stochastic Gradient Descent"]
    classifiers = [KNeighborsClassifier(3), DecisionTreeClassifier(max_depth=5), RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight = {0: 1, 1:10}), AdaBoostClassifier(), BaggingClassifier(), linear_model.LogisticRegression(), SGDClassifier(loss="modified_huber", penalty="l2")]

    #start k-fold
    for x in range(0, 1):
        print "Split: " + str(x)
        train_data, test_data = train_test_split(data, test_size=.3) 
 
        #conditional mean imputation
        for col in test_data.columns.tolist():
            test_data[col] = test_data[col].fillna(value=data[col].mean())
        for col in train_data.columns.tolist():
            train_data[col] = train_data[col].fillna(value=train_data[col].mean())

        # define xs, y
        colList = data.columns.tolist()
        colList.remove(DV)
        x_train, x_test = train_data.loc[:,colList], test_data.loc[:,colList]
        y_train, y_test = train_data.loc[:,DV], test_data.loc[:,DV]

        #loop through classifiers, get predictions, scores, make miss-classified tree
        clf_results = {}
        clf_results[x] = {}
        for name, clf in zip(names, classifiers):
            preds, probs, train_time, test_time = fitClf(clf, x_train, y_train, x_test)
            clf_results[x][name] = getScores(clf_results, x, name, clf, y_test, preds, x_test, train_time, test_time)
            #findMisClf(test_data, x_test, y_test, preds, name)

    #print clf_results 
    print "End"


main()
