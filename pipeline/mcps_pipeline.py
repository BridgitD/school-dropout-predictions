######################
#                    #
#  MCPS Pipeline     #
#  Bridgit Donnelly  #
#                    #
######################

import pandas as pd
import numpy as np
import pipeline as ml


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

def clean_data(df):

    print "Cleaning data..."

    ################################
    ## DROP UNNECESSARY VARIABLES ##
    ################################

    print "Dropping unnecessary variables..."

    variables_to_drop = ['g6_tardyr','g6_school_name', 'g7_school_name', 'g8_school_name', 'g9_school_name', 'g10_school_name', 'g11_school_name', 'g12_school_name','g6_year', 'g6_gradeexp', 'g6_grade', 'g6_wcode', 'g7_year', 'g7_gradeexp', 'g7_grade', 'g7_wcode', 'g8_year', 'g8_gradeexp', 'g8_grade', 'g8_wcode', 'g9_year', 'g9_gradeexp', 'g9_grade', 'g9_wcode', 'g10_year', 'g10_gradeexp', 'g10_grade', 'g10_wcode', 'g11_year', 'g11_gradeexp', 'g11_grade', 'g11_wcode', 'g12_year', 'g12_gradeexp', 'g12_grade', 'g12_wcode']
    for v in variables_to_drop:
        df.drop(v, axis=1, inplace=True)

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

    ###################################
    ## CREATE DUMMY VARIABLE COLUMNS ##
    ###################################

    pd.get_dummies(df, dummy_na=True)

    ## Save clean version
    ml.print_to_csv(df, '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data.csv')


def impute_data(dataset):

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

    ## Fill missing MSAM data
    msam_cols = ['g6_g6msam', 'g7_g7msam', 'g8_g8msam', 'g9_g8msam']
    
    #Recode existing data
    for col in msam_cols:
        for index, row in df.iterrows():
            if row[col] == "Advanced":
                df.ix[index, col] = 2
            elif row[col] == "Proficient":
                df.ix[index, col] = 1
            elif row[col] == "Basic":
                df.ix[index, col] = 0

    # Impute missing data
    ml.replace_with_mean(df, msam_cols)

    ############################
    ## IMPUTE BEHAVIORAL DATA ##
    ############################

    print "Impute missing behavioral data..."

    ## Fill missing behavioral data -- use mean imputation for now
    behavioral_cols = ['g6_absrate', 'g6_nsusp','g7_absrate', 'g7_tardyr', 'g7_nsusp', 'g8_absrate', 'g8_tardyr', 'g8_nsusp', 'g9_absrate', 'g9_nsusp', 'g10_absrate', 'g10_nsusp', 'g11_absrate', 'g11_nsusp','g12_absrate', 'g12_nsusp']
    ml.replace_with_mean(df, behavioral_cols)

    ############################
    ## IMPUTE ENROLLMENT DATA ##
    ############################

    print "Imputing missing enrollment data..."

    ## Fill missing enrollment data
    print "Fixing mobility columns..."
    mobility_cols = ['g10_retained', 'g6_mobility', 'g7_mobility', 'g8_mobility', 'g9_mobility','g10_mobility', 'g11_mobility', 'g12_mobility']
    # Includes g10_retained because it's coded as 0/1 already
    ml.replace_with_mean(df, mobility_cols)


    #########################
    ## IMPUTE DROPOUT DATA ##
    #########################

    print "Impute missing droput information..."

    ## Fill missing dropout information with 0
    dropout_vars = ['g6_dropout', 'g7_dropout', 'g8_dropout', 'g9_dropout', 'g10_dropout', 'g11_dropout', 'g12_dropout']
    ml.replace_with_value(df, dropout_vars, [0,0,0,0,0,0,0])

    #variables = list(df.columns.values)
    #print variables

    summary = ml.summarize(df)
    print summary.T
    #ml.print_to_csv(summary.T, 'updated_summary_stats_vertical.csv')

    ml.print_to_csv(df, '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/imputed_data.csv')
    print "Done!"

#-------------------------------------------------------

if __name__ == '__main__':

    dataset = "/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/cohort1_all.csv"

    #df = summarize_data(dataset)
    df = ml.read_data(dataset)
    clean_data(df)

    clean_dataset = '/mnt/data2/education_data/mcps/DATA_DO_NOT_UPLOAD/clean_data.csv'
    #impute_data(clean_dataset)
