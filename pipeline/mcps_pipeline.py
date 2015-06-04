######################
#                    #
#  MCPS Pipeline     #
#  Bridgit Donnelly  #
#                    #
######################

import pandas as pd
import numpy as np
import pipeline as ml
import sys
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def summarize_data(dataset):

    ####################################
    ## RUN INITIAL SUMMARY STATISTICS ##
    ####################################
    print "Running summary statistics..."

    ml.getSumStats(df)
    ## HISTOGRAMS NEED TO BE ADDED ##

    #ml.summarize_dataset(dataset)
    #for v in variables:
    #    ml.summary_statistics(v, dataset, 5, 10)


def clean_data(df, cohort):

    print "Cleaning data..."

    ################################
    ## DROP UNNECESSARY VARIABLES ##
    ################################

    print "Dropping unnecessary variables..."

    if cohort == 1:
        print "for cohort 1..."
        variables_to_drop = ['g6_tardyr','g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_gradeexp', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_gradeexp', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_gradeexp', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_gradeexp', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_gradeexp', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_gradeexp', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_gradeexp', 'g12_grade', 'g12_wcode']
        df.drop(variables_to_drop, axis=1, inplace=True)
        
    elif cohort == 2:
        print "for cohort 2..."
        variables_to_drop = ['g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_grade', 'g12_wcode']
        df.drop(variables_to_drop, axis=1, inplace=True)

    #######################
    ## COMBINE VARIABLES ##
    #######################

    ## Create single column for birth year
    print "Correcting birthdays..."

    df['birthday'] = df['g11_byrmm']
    birthday_cols = ['g12_byrmm', 'g10_byrmm', 'g9_byrmm', 'g8_byrmm', 'g7_byrmm', 'g6_byrmm']
    for c in birthday_cols:
        ml.replace_if_missing(df, 'birthday', c)
        df.drop(c, axis=1, inplace=True)
    
    df['birth_year'] = df.loc[:,'birthday'].astype(str, copy=False)[:4]
    df['birth_month'] = df.loc[:,'birthday'].astype(str, copy=False)[4:]
    df.drop('birthday')

    ## Create single column for gender
    print "Correcting gender..."

    df['gender'] = df['g11_gender']
    gender_cols = ['g12_gender', 'g11_gender', 'g10_gender', 'g9_gender', 'g8_gender', 'g7_gender', 'g6_gender']
    for c in gender_cols:
        ml.replace_if_missing(df, 'gender', c)
        df.drop(c, axis=1, inplace=True)


    ################
    ## CLEAN DATA ##
    ################

    print "Cleaning data..."
    retained_cols = ['g11_retained', 'g12_retained', 'g9_newmcps', 'g10_newmcps', 'g11_newmcps', 'g12_newmcps', 'g9_newus', 'g10_newus', 'g11_newus', 'g12_newus']

    for col in retained_cols:
        df[col] = df[col].notnull()


    ###############################
    ## CREATE MISSING DATA FLAGS ##
    ###############################

    print "Creating missing data flags..."

    ## Create flag if a given student is missing a year's worth of data
    grade_id = ['g6_pid', 'g7_pid', 'g8_pid', 'g9_pid', 'g10_pid', 'g11_pid', 'g12_pid']
    year = 6
    for g in grade_id:
        col_name = 'g' + str(year) + '_missing'
        df[col_name] = df[g].isnull()
        df.drop(g, axis=1, inplace=True)
        year+=1

    return_file = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/predummy_data_cohort' + str(cohort) + '.csv'
    ml.print_to_csv(df, return_file)

    return df


def deal_with_dummies(df, cohort):

    if isinstance(df, str):
        df = ml.read_data(df)
    
    ###################################
    ## CREATE DUMMY VARIABLE COLUMNS ##
    ###################################
    print "Creating dummy variables..."

    school_ids = [col for col in df.columns if 'school_id' in col]
    df[school_ids] = df.loc[:,school_ids].astype(str, copy=False)

    string_cols = list(df.select_dtypes(include=['object']))
    
    dummys = pd.get_dummies(df[string_cols], dummy_na=True)
    df = pd.concat([df, dummys], axis=1)
    
    df.drop(string_cols, axis=1, inplace=True)

    ## Save clean version
    return_file = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data_cohort' + str(cohort) + '.csv'
    ml.print_to_csv(df, return_file)

    return df


def choose_data(df, grade):
    print "Choosing data..."

    if isinstance(df, str):
        df = ml.read_data(df)

    #Find columns to use
    print "Choosing columns..."
    all_columns = list(df.columns.values)
    cols_to_use = []

    i = grade
    prefixes = []

    while i <= 12:
        prefixes.append('g' + str(i))
        i+=1

    for col in all_columns:
        for p in prefixes:
            if not col.startswith(p):
                if col not in cols_to_use:
                    cols_to_use.append(col)

    for index, val in enumerate(cols_to_use):
        if val.startswith('Unnamed'):
            cols_to_use.pop(index)

    dv = 'g' + str(grade) + '_dropout'

    #Find rows to use
    print "Choosing rows..."
    data9 = df[df['g6_dropout'] !=1]
    data9 = data9[data9['g7_dropout'] !=1]
    data9 = data9[data9['g8_dropout'] !=1]
    data10 = data9[data9['g9_dropout'] !=1]
    data11 = data10[data10['g10_dropout'] !=1]
    data12 = data11[data11['g11_dropout'] !=1]

    if grade == 9:
        return data9, data9[dv], data9[cols_to_use]
    elif grade == 10:
        return data10, data10[dv], data10[cols_to_use]
    elif grade == 11:
        return data11, data11[dv], data11[cols_to_use]
    elif grade == 12:
        return data12, data12[dv], data12[cols_to_use]


def impute_data(df):

    if isinstance(df, str):
        df = ml.read_data(df)

    #########################
    ## IMPUTE MISSING DATA ##
    #########################
    print "Imputing missing data..."

    #change msam to missing is msam_NA==1
    nanList =  ['g6_g6msam_nan', 'g7_g7msam_nan', 'g8_g8msam_nan', 'g9_g8msam_nan']
    varList = [[ 'g6_g6msam_Advanced', 'g6_g6msam_Basic', 'g6_g6msam_Proficient'], ['g7_g7msam_Advanced', 'g7_g7msam_Basic', 'g7_g7msam_Proficient'], ['g8_g8msam_Advanced', 'g8_g8msam_Basic', 'g8_g8msam_Proficient'],['g9_g8msam_Advanced', 'g9_g8msam_Basic', 'g9_g8msam_Proficient']]
    for x in range(0,len(nanList)):
        nacol = nanList[x]
        colList = varList[x]
        for col in colList:
            df.loc[df[nacol] == 1, col] = np.nan 


    #pred missing data using any available data
    wordList = ['absrate', 'mapr', 'msam_Advanced', 'msam_Basic', 'msam_Proficient', 'mobility', 'nsusp', 'mpa', 'tardyr', 'psatm', 'psatv', 'retained']
    for word in wordList:
        colList = [col for col in df.columns if word in col]
        rowMean = df[colList].mean(axis=1)
        for col in colList:
                df[col].fillna(rowMean, inplace=True)


    '''
    ############################
    # IMPUTE NEIGHBORHOOD DATA #
    ############################

    print "Imputing missing school neighborhood data..."

    ## Fill missing school neighborhood data
    print "Fixing neighborhood columns..."
    neighborhood_cols = ['suspensionrate',  'mobilityrateentrantswithdra',  'attendancerate',   'avg_class_size',   'studentinstructionalstaffratio',   'dropoutrate',  'grade12documenteddecisionco',  'grade12documenteddecisionem',  'grade12documenteddecisionmi',  'grad12docdec_col_emp', 'graduationrate',   'studentsmeetinguniversitysyste',   'Est_Households_2012',  'Est_Population_2012',  'Med_Household_Income_2012',    'Mean_Household_Income_2012',   'Pop_Below_Poverty_2012',   'Percent_Below_Poverty_2012',   'Pop_Under18_2012', 'Under18_Below_Poverty_2012',   'Under18_Below_Poverty_Percent_2012',   'Housholds_on_Food_stamps_with_Children_Under18_2012',  'Housholds_Pop_on_Food_Stamps_2012',    'Pop_BlackAA_2012', 'Pop_White_2012',   'Bt_18_24_percent_less_than_High_School_2012',  'Bt_18_24_percent_High_School_2012',    'Bt_18_24_percent_Some_College_or_AA_2012', 'Bt_1824_percent_BA_or_Higher_2012',    'Over_25_percent_less_than_9th_grade_2012', 'Over_25_percent_9th_12th_2012',    'Over_25_percent_High_School_2012', 'Over_25__percent_Some_College_No_Deg_2012',    'Over_25_percent_AA_2012',  'Over_25_percent_Bachelors_2012',   'Over_25_percent_Graduate_or_Professionals_2012']
    ml.replace_with_mean(df, neighborhood_cols)
    '''

    #summary = ml.summarize(df)
    #print summary.T
    #ml.print_to_csv(summary.T, 'updated_summary_stats_vertical.csv')

    return_file = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/imputed_data_cohort' + str(cohort) + '.csv'
    ml.print_to_csv(df, return_file)
    print "Done!"
    return df

def fit_models(df, X, y):
   # Classifiers to test
    classifiers = [('logistic_regression', LogisticRegression())] 
                   #('k_nearest_neighbors', KNeighborsClassifier()),
                   #('decision_tree', DecisionTreeClassifier()),
                   #('SVM', LinearSVC()),
                   #('random_forest', RandomForestClassifier()),
                   #('boosting', GradientBoostingClassifier()),
                   #('bagging', BaggingClassifier())]

    ml.build_classifiers(df, X, y, classifiers)
    #ml.test_classifier(df, X, y, classifiers)
 

#-------------------------------------------------------

if __name__ == '__main__':

    ## ORIGINAL DATASETS
    dataset = "/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort1_all.csv"
    test = "/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort2_all.csv"

    ## LOAD DATA
    df = ml.read_data(dataset)
    test = ml.read_data(test)

    ## RUN SUMMARY STATISTICS
    summarize_data(df)
    summarize_data(test)

    ## CLEAN DATA
    print "Cleaning Cohort 1..."
    predummy_data_cohort1 = clean_data(df, 1)
    
    print "Cleaning Cohort 2..."
    predummy_data_cohort2 = clean_data(test, 2)

    deal_with_dummies(non_dummy_cohort1, 1)
    deal_with_dummies(non_dummy_cohort2, 2)

    ## TRAINING DATA: CHOOSE SUBSET
#    clean_cohort1 = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data_cohort1.csv'
    df, y, X = choose_data(clean_cohort1, 12)
 
    ## TRAINING DATA: IMPUTATION
    df = impute_data(df, 1)

    ## TRAINING DATA: START K-FOLD WITH CORRECT DATA
#    imputed_dataset = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/imputed_data.csv'
#    df = ml.read_data(imputed_dataset)
#    df, y, X = choose_data(df, 12)

    ## TRAINING DATA: FEATURE GENERATION

    ## TRAINING DATA: MODEL FITTING
    fit_models(df, X, y)

    ## TRAINING DATA: ID MISCLASSIFICATION
    #clean_dataset = 'data/clean_data.csv'
    #clean_dataset = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data.csv'
    #impute_data(clean_dataset, 'cohort1')
    #impute_data(clean_dataset, 'cohort2')
