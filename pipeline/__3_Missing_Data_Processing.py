"""
Joshua Mausolf - CAPP 30254 Assignment pa3.

In this python module I pre-process the data by filling in missing values.
"""

import sys, os
import csv
import pandas as pd 
import re


#_____________ PART 3A _______________________________________________ #

# Pre-process values by filling in missing values.

## I made the choice to round to the nearest integer values to match the formatting
## for the existing database.


def camel_to_snake(column_name):
    """
    Converts a string that is camelCase into snake_case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()




#______________Impute Mean_________________#


def impute_mean(variable, data_in, data_out):

    #Define Data
    data = pd.read_csv(data_in, index_col=0)
    data.columns = [camel_to_snake(col) for col in data.columns]

    #Get Row Index for Variable
    number = (data.columns.get_loc(str(variable)))+1

    #Generate Mean
    m_var = data[str(variable)].mean()
    meanVar = int(round(m_var))

    in_file = open(data_in, 'rU')
    reader = csv.reader(in_file)
    out_file = open(data_out, "w")
    writer = csv.writer(out_file)
    for row in reader:
        #Monthly_income = row[6]
        variable_observation = row[number]
        if variable_observation == '':
            row[number] = meanVar
            writer.writerow(row)
        elif variable_observation == 'NA':
            row[number] = meanVar
            writer.writerow(row)
        else:
            writer.writerow(row)
    in_file.close()
    out_file.close()


#Unhash to run
impute_mean('monthly_income', 'data/cs-training.csv', 'data/cs-training#3A.csv')
impute_mean('number_of_dependents', 'data/cs-training#3A.csv', 'data/cs-training#3B.csv')







