

import pandas as pd
import math
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

constant = 1/math.sqrt(2 * 3.14159)

def compute_mean_variance(df, attribute_index, class_index):
    temp_df = df[df[10].isin([class_index])]
    return (temp_df[attribute_index].mean(skipna=True), temp_df[attribute_index].var(skipna=True));


def populate_Mean_Variance(df):
    mean_variance = dict(); # dictionary will be of the format element name followed by tuple(mean, variance)
    for attribute_index in range(1, 10): #since there are 9 attributes
        for class_index in range(1, 8):
            if (mean_variance.get(attribute_index, -1) == -1):
                mean_variance[attribute_index] = list();
            tuple_returned = compute_mean_variance(df, attribute_index, class_index);
            mean_variance[attribute_index].append( tuple_returned);
    return mean_variance

"""
    classIndex : from 0 to 7
    attributeIndex : from 1 to 10
    mean_variance : dictionary for each of the 9 attributes and corresponding 7 (mean, variance) tuples
"""
def gaussian_probability(dataFrame, classIndex, attributeIndex, mean_variance):

    class_mean = float(mean_variance[attributeIndex][classIndex][0])
    class_variance = float(mean_variance[attributeIndex][classIndex][1])
    x = float(dataFrame.loc[:, attributeIndex ])
    numerator = float(pow( x - class_mean, 2))
    denominator = float(2.0 * class_variance * class_variance)
    if (class_variance > 0.0 or class_variance < 0.0):
        return ( float(constant/class_variance) * float(pow(2.71828 , -(numerator/denominator))) )
    return 0

def compute_probability(df, mean_variance, class_numbers):
    #print "Received DF : ", df
    probabilites = [];
    for i in range(0, 7): #For each 7 classes
        result = 0.0
        for j in range(1, 10): #since there are 9 attributes
    #        print "Computing probability for class = ", i + 1 , " and attribute index = ", j
            result = float(result) +  float(class_numbers[i]/float(sum(class_numbers))) * float(gaussian_probability(df, i, j, mean_variance));
        probabilites.append(result)
    return probabilites

def processNaive(training_data_file, testing_data_file):
    df = pd.read_table(training_data_file,
                       delimiter=',',
                       header=None);

    class_numbers = []; #Frequency of each class type
    for i in range(1, 8): #Since we want till 7
        class_numbers.append(len(df[df[10].isin([i])]))

    """
        This function will compute mean and variance for each class.
        Now in this case there are 7 class types and 9 attributes. Therefore
        this is a hashmap of 9 keys each containing 7 tuples of two values(mu and sigma)
    """
    mean_variance = populate_Mean_Variance(df)

    #Read test data
    test_df = pd.read_table(testing_data_file,
                       delimiter=',',
                       header=None);

    total_test_data = len(test_df)
    accuracy = 0;

    for i in range(0, len(test_df)):
        probabilites = compute_probability(test_df.loc[[i], :], mean_variance, class_numbers) #Sending the first row and rrequired data
        max_probability = max(probabilites);
        predicted_class = probabilites.index(max_probability) + 1

        actual_class = int(test_df.loc[i, 10])

        if (predicted_class == actual_class):
            accuracy = accuracy + 1

    return float(accuracy/float(total_test_data)) * 100.0

