
import NaiveBayes
import kNNClassifier
import sys

if __name__ == '__main__':
    training_data_file = sys.argv[1]
    testing_data_file = sys.argv[2]

    print "#############################################################################################"
    print "                                  Naive Bayes Algo"
    print "#############################################################################################"
    print "Test Data accuracy   : " , str(NaiveBayes.processNaive(training_data_file, testing_data_file)) + "%"
    print "Train Data accuracy  : " , str(NaiveBayes.processNaive(training_data_file, training_data_file)) + "%"

    print "#############################################################################################"
    print "                                  kNN Classfier Algo(Test Data Accuracy)"
    print "#############################################################################################"
    kNNClassifier.processkNN(training_data_file, testing_data_file)

    print "#############################################################################################"
    print "                                  kNN Classfier Algo(Training Data Accuracy)"
    print "#############################################################################################"
    kNNClassifier.processkNN(training_data_file, training_data_file)

