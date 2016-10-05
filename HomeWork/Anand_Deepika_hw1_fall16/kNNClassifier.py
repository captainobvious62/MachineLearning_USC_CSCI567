

import sys
import pandas as pd
import numpy as np
import math

def readData(dataFile):
    return pd.read_table(dataFile,
                       delimiter=',',
                       header=None);

def euclideanDistance(train_instance, test_instance):
    dist = 0
    for i in range(0, len(train_instance)-1):
        dist = dist + pow( (train_instance[i] - test_instance[i]), 2)

    return (math.sqrt(dist), int(train_instance[9]))

def manhattanDistance(train_instance, test_instance):
    dist = 0
    for i in range(0, len(train_instance)-1):
        dist = dist + abs(train_instance[i]-test_instance[i])
    return (dist, int(train_instance[9]))

def trimData(distances, k):
    if (k == 1):
        return int(distances[0][1])
    else:
        result = dict()

        for distance in distances[0:k+1]: #Select top k classes
                result[distance[1]] = result.get(distance[1], 0) + 1

        #Result is of the form class_name, class_count
        max_class = ''
        max_class_size = -1
        for (key,value) in result.iteritems():
            if value > max_class_size:
                max_class = key
                max_class_size = value

        # All mappings of distance, class where class count is same
        number_of_max_class = filter(lambda x: x[1] == max_class_size, result.iteritems())

        if (len(number_of_max_class) == 1 ): #If we have a clear winner
            return max_class
        else:
            min_candidate = sys.maxint
            min_candidate_class = ''

            for item in number_of_max_class: #should be mapping of class_name and count
                class_name = item[0]
                max_candidate_class = [ item for item in distances[0:k+1] if item[1] == class_name]

                result = sorted(max_candidate_class, key=lambda x: x[0]) #sort according to distance
                if (result[0][0] < min_candidate):
                    min_candidate = result[0][0]
                    min_candidate_class = class_name
            return min_candidate_class

def get_distance(training_df, test_instance):

    result = []
    L1_distances = list()
    L2_distances = list()
    for train_instance in np.array(training_df):
        L1_distances.append( euclideanDistance(train_instance, test_instance) ) #Call L1 first
        L2_distances.append( manhattanDistance(train_instance, test_instance) )

    #Get k minimum distances
    #Do the voting
    #Return class with maximum votes
    L1_distances.sort()
    for i in [1, 3, 5, 7]:
        result.append( trimData(L1_distances, i))

    L2_distances.sort()
    for i in [1, 3, 5, 7]:
        result.append( trimData(L2_distances, i))
    return result

def run_prediction(training_df, testing_df):
    accuracy_denom = len(testing_df); #Becuase for each case we will remain [1, 3, 5, 7] and L1, L2 combination
    result = []
    for test_instance in np.array(testing_df):
        predicted_classes = get_distance(training_df, test_instance)
        actual_class = int(test_instance[9])
        predicted_classes.append(actual_class)

        result.append(predicted_classes)

    trigger = 1
    print "      ",  "     ", "     L1" , "           |  ",  "   L2"
    for i in range(0, 4): #since we will have 8 combinations
        countA = 0
        countB = 0

        for item in result:
            if (item[i] == item[-1]):
                countA = countA + 1
            if (item[i + 4] == item[-1]):
                countB = countB + 1

        print " k = ", i + trigger, "     ", float(countA/float(accuracy_denom))*100, "%" , "  |  ",  float(countB/float(accuracy_denom))*100, "%"
        trigger = trigger + 1

def kNN(training_df, testing_df):

    mean = list(training_df.mean())
    mean[-1] = 0

    var = list(training_df.std())
    var[-1] = 1

    #Normalize both training set and testing set
    training_df = (training_df - mean)/var
    testing_df = (testing_df- mean)/var

    run_prediction(training_df, testing_df)

    #print testing_df
def processkNN(training_data_file, testing_data_file):

    trainig_df = readData(training_data_file)
    testing_df = readData(testing_data_file)

    kNN(trainig_df.loc[:, 1:], testing_df.loc[:, 1:])
