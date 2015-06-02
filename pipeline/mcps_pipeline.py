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
from sklearn.ensemble import RandomForestClassifier#, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve

def summarize_data(dataset):

    ###############
    ## LOAD DATA ##
    ###############

    print "Loading data..."

    df = ml.read_data(dataset)
    variables = list(df.columns.values)
    #print variables

    ####################################
    ## RUN INITIAL SUMMARY STATISTICS ##
    ####################################
    print "Running summary statistics..."

    ml.summarize_dataset(dataset)
    for v in variables:
        ml.summary_statistics(v, dataset, 5, 10)

    return df

def clean_data(df, cohort):

    print "Cleaning data..."

    ################################
    ## DROP UNNECESSARY VARIABLES ##
    ################################

    print "Dropping unnecessary variables..."

    if cohort == 1:
        print "for cohort 1..."
        variables_to_drop = ['g6_tardyr','g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_gradeexp', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_gradeexp', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_gradeexp', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_gradeexp', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_gradeexp', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_gradeexp', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_gradeexp', 'g12_grade', 'g12_wcode']
        for v in variables_to_drop:
            df.drop(v, axis=1, inplace=True)

    elif cohort == 2:
        print "for cohort 2..."
        variables_to_drop = ['g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_grade', 'g12_wcode']
        for v in variables_to_drop:
            df.drop(v, axis=1, inplace=True)

    else:
        pass

    #######################
    ## COMBINE VARIABLES ##
    #######################

    ## Create single column for birth year
    print "Correcting birthdays..."

    df['birthday'] = df['g11_byrmm']
    birthday_cols = ['g12_byrmm', 'g11_byrmm', 'g10_byrmm', 'g9_byrmm', 'g8_byrmm', 'g7_byrmm', 'g6_byrmm']
    for c in birthday_cols:
        ml.replace_with_other_col(df, 'birthday', c)
        df.drop(c, axis=1, inplace=True)
    #print ml.summarize(df['birthday'])

    ## Create single column for gender
    print "Correcting gender..."

    df['gender'] = df['g11_gender']
    gender_cols = ['g12_gender', 'g11_gender', 'g10_gender', 'g9_gender', 'g8_gender', 'g7_gender', 'g6_gender']
    for c in gender_cols:
        ml.replace_with_other_col(df, 'gender', c)
        df.drop(c, axis=1, inplace=True)
    #print df['gender'].value_counts()


    ################
    ## CLEAN DATA ##
    ################

    print "Cleaning data..."
    retained_cols = ['g11_retained', 'g12_retained', 'g9_newmcps', 'g10_newmcps', 'g11_newmcps', 'g12_newmcps', 'g9_newus', 'g10_newus', 'g11_newus', 'g12_newus']
    for col in retained_cols:
        for index, row in df.iterrows():
            if pd.isnull(row[col]):
                df.ix[index, col] = 0
            else:
                df.ix[index, col] = 1
        df[col] = df[col].astype(int)


    ###############################
    ## CREATE MISSING DATA FLAGS ##
    ###############################

    print "Creating missing data flags..."

    ## Create flag if a given student is missing a year's worth of data
    grade_id = ['g6_pid', 'g7_pid', 'g8_pid', 'g9_pid', 'g10_pid', 'g11_pid', 'g12_pid']
    year = 6
    for g in grade_id:
        col_name = 'g' + str(year) + '_missing'
        for index, row in df.iterrows():
            if pd.isnull(row[g]):
                df.ix[index, col_name] = 1
            else:
                df.ix[index, col_name] = 0
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

    string_cols = list(df.select_dtypes(include=['object']))
    
    df = ml.get_dummys(df, string_cols, dummy_na=True)
    for col in string_cols:
        df.drop(col, axis=1, inplace=True)

    ## Save clean version
    return_file = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data_cohort' + str(cohort) + '.csv'
    ml.print_to_csv(df, return_file)

    return df


def choose_data(df, grade):

    if isinstance(df, str):
        df = ml.read_data(df)

    #Find columns to use
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
    #print cols_to_use

    #Find rows to use
    data9 = df[df['g6_dropout'] !=1]
    data9 = data9[data9['g7_dropout'] !=1]
    data9 = data9[data9['g8_dropout'] !=1]
    data10 = data9[data9['g9_dropout'] !=1]
    data11 = data10[data10['g10_dropout'] !=1]
    data12 = data11[data11['g11_dropout'] !=1]

    if grade == 9:
        return dv, cols_to_use, data9
    elif grade == 10:
        return dv, cols_to_use, data10
    elif grade == 11:
        return dv, cols_to_use, data11
    elif grade == 12:
        return dv, cols_to_use, data12


def impute_data(df, cohort):

    if isinstance(df, str):
        df = ml.read_data(df)

    ##########################
    ## IMPUTE ACADEMIC DATA ##
    ##########################

    print "Impute missing academic information..."

    ## Fill missing school data -- use mean imputation for now
    school_vars = ['g6_school_id', 'g7_school_id', 'g8_school_id', 'g9_school_id', 'g10_school_id', 'g11_school_id', 'g12_school_id']
    ml.replace_with_mean(df, school_vars)

    ## Fill missing grade and test score information -- use mean imputation for now
    grades_tests = ['g6_q1mpa', 'g6_q2mpa', 'g6_q3mpa', 'g6_q4mpa', 'g6_g6mapr','g7_q1mpa', 'g7_q2mpa', 'g7_q3mpa', 'g7_q4mpa', 'g7_g7mapr', 'g8_q1mpa', 'g8_q2mpa', 'g8_q3mpa', 'g8_q4mpa', 'g8_g8mapr', 'g9_q1mpa', 'g9_q2mpa', 'g9_q3mpa', 'g9_q4mpa', 'g9_g8mapr', 'g10_q1mpa', 'g10_q2mpa', 'g10_q3mpa', 'g10_q4mpa', 'g10_psatv', 'g10_psatm', 'g11_q1mpa', 'g11_q2mpa', 'g11_q3mpa', 'g11_q4mpa', 'g11_psatv', 'g11_psatm', 'g12_q1mpa', 'g12_q2mpa', 'g12_q3mpa', 'g12_q4mpa', 'g12_psatv', 'g12_psatm']
    ml.replace_with_mean(df, grades_tests)

    ## Fill in missing id with dummy
    #ml.replace_with_value(df, 'id', 0)

    ## Fill missing MSAM data
    g6_msam = ['g6_g6msam_Advanced','g6_g6msam_Basic','g6_g6msam_Proficient']
    ml.replace_dummy_null_mean(df, 'g6_g6msam_nan', g6_msam)

    if cohort == 1:
        g7_msam = ['g7_g7msam_Advanced','g7_g7msam_Basic','g7_g7msam_Proficient']
        ml.replace_dummy_null_mean(df, 'g7_g7msam_nan', g7_msam)
    elif cohort == 2:
        g7_msam = ['g7_g7msam_ ','g7_g7msam_1','g7_g7msam_2', 'g7_g7msam_3']
        ml.replace_dummy_null_mean(df, 'g7_g7msam_nan', g7_msam)

    g8_msam = ['g8_g8msam_Advanced','g8_g8msam_Basic','g8_g8msam_Proficient']
    ml.replace_dummy_null_mean(df, 'g8_g8msam_nan', g8_msam)

    g9_msam = ['g9_g8msam_Advanced','g9_g8msam_Basic','g9_g8msam_Proficient']
    ml.replace_dummy_null_mean(df,'g9_g8msam_nan', g9_msam)

    
    ############################
    ## IMPUTE BEHAVIORAL DATA ##
    ############################

    print "Impute missing behavioral data..."

    ## Fill missing behavioral data -- use mean imputation for now
    behavioral_cols = ['g6_absrate', 'g6_nsusp','g7_absrate', 'g7_tardyr', 'g7_nsusp', 'g8_absrate', 'g8_tardyr', 'g8_nsusp', 'g9_absrate', 'g9_nsusp', 'g10_absrate', 'g10_nsusp', 'g11_absrate', 'g11_nsusp','g12_absrate', 'g12_nsusp']
    ml.replace_with_mean(df, behavioral_cols)

    ## Fill in missing birthday data
    #ml.replace_with_mean(df, 'birthday')

    ############################
    ## IMPUTE ENROLLMENT DATA ##
    ############################

    print "Imputing missing enrollment data..."

    ## Fill missing enrollment data
    print "Fixing mobility columns..."
    mobility_cols = ['g10_retained', 'g6_mobility', 'g7_mobility', 'g8_mobility', 'g9_mobility', 'g9_retained','g10_mobility', 'g11_mobility', 'g12_mobility', 'birthday']
    # Includes g10_retained because it's coded as 0/1 already
    ml.replace_with_mean(df, mobility_cols)


    #########################
    ## IMPUTE DROPOUT DATA ##
    #########################

    print "Impute missing droput information..."

    ## Fill missing dropout information with 0
    dropout_vars = ['g6_dropout', 'g7_dropout', 'g8_dropout', 'g9_dropout', 'g10_dropout', 'g11_dropout', 'g12_dropout', 'dropout']
    ml.replace_with_value(df, dropout_vars, [0,0,0,0,0,0,0,0])

    #variables = list(df.columns.values)
    #print variables


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

    #ml.print_to_csv(df, 'data/imputed_data.csv')
    ml.print_to_csv(df, '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/imputed_data.csv')
    print "Done!"


#-------------------------------------------------------

if __name__ == '__main__':

    ## ORIGINAL DATASETS
#    dataset = "/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort1_all.csv"
#    test = "/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort2_all.csv"

    ## RUN SUMMARY STATISTICS
#    df = summarize_data(dataset)

    ## CLEAN DATA
#    df = ml.read_data(dataset)
#    print "Cleaning Cohort 1..."
#    predummy_data_cohort1 = clean_data(df, 1)
    
#    print "Cleaning Cohort 2..."
#    predummy_data_cohort2 = clean_data(df, 2)

#    non_dummy_cohort1 = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/predummy_data_cohort1.csv'
#    non_dummy_cohort2 = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/predummy_data_cohort2.csv'
#    deal_with_dummies(non_dummy_cohort1, 1)
#    deal_with_dummies(non_dummy_cohort2, 2)

    ## TRAINING DATA: CHOOSE SUBSET
    clean_cohort1 = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data_cohort1.csv'
    grade = sys.argv[1]
    dv, cols_to_use, df = choose_data(clean_cohort1, grade)
    print df[cols_to_use]

    ## TRAINING DATA: IMPUTATION
    #impute_data(clean_dataset, 1)

    ## TRAINING DATA: START K-FOLD WITH CORRECT DATA
    #imputed_dataset = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/imputed_data.csv'
    #df = ml.read_data(imputed_dataset)
    #y, X =  choose_data(df, 12)

    ## TRAINING DATA: FEATURE GENERATION

    ## TRAINING DATA: MODEL FITTING
    # Classifiers to test
    #classifiers = [('logistic_regression', LogisticRegression())]
                    #('k_nearest_neighbors', KNeighborsClassifier()),
                    #('decision_tree', DecisionTreeClassifier())]
                    #('SVM', LinearSVC()),
                    #('random_forest', RandomForestClassifier()),
                    #('boosting', GradientBoostingClassifier()),
                    #('bagging', BaggingClassifier())]

    #ml.test_classifier(df, X, y, classifiers)

    ## TRAINING DATA: ID MISCLASSIFICATION
    #clean_dataset = 'data/clean_data.csv'
    #clean_dataset = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data.csv'
    #impute_data(clean_dataset, 'cohort1')
    #impute_data(clean_dataset, 'cohort2')
