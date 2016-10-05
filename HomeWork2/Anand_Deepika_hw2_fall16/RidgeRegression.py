
import numpy as np
import pandas as pd
import time

def y_hat(x, w):
    return x.dot(w)

def gradient_descent(x, y, w, max_iter, lambdaValue, alpha = 0.05 ):
    N = y.shape[0]

    for i in range(0, max_iter):
        #print " Running iteration : ", i
        XW = y_hat(x, w)
        wTw = w.T.dot(w)

        gradient = np.dot(x.T, XW - y) / N + float(lambdaValue) * wTw

        w = w - alpha * gradient

    return w

def runGenericRidgeRegression(df, lambdaValue):

    numpyMatrix = df.as_matrix()
    x = np.array(numpyMatrix[:, 0:numpyMatrix.shape[1]-1]) # Select all rows all columns but the last column itself

    (m,n) = x.shape
    one_column = np.ones(shape=(m, 1)) # Create a matrix with m rows and 1 column

    x = np.c_[ one_column, x ] # now x conatins values of the form 1, x1, x2,...xn
    m,n=np.shape(x)

    y = np.array(numpyMatrix[:, numpyMatrix.shape[1]-1:numpyMatrix.shape[1]]) # this is the target value
    w = np.array([np.ones(n)]).T
    w_updated = gradient_descent(x, y, w, 7, lambdaValue)

    return w_updated

def display(results):
    print "%%%%%%%%%Displaying ridge regression result for lambda = (0.01, 0.1, 1.0)%%%%%%%%%%%%%"
    time.sleep(2)
    print "Lambda----------------------> 0.01     0.1       1.0"
    print "MSE_Test Data               ", round(results[0], 3), "   ", round(results[2], 3), "    ", round(results[4], 3)
    print "MSE_Train Data              ", round(results[1], 3), "   ", round(results[3], 3), "    ", round(results[5], 3)

def startRidgeRegression(training_df, testing_df, start, stop, runTestCasesGeneric, runOnTrainData = True):

    results = []
    lambdaValue = start;
    while(lambdaValue <= stop):
        w_updated = runGenericRidgeRegression(training_df, lambdaValue)
        residue, MSE = runTestCasesGeneric(w_updated, testing_df)
        results.append(MSE)
        if (runOnTrainData == True):
            residue, MSE = runTestCasesGeneric(w_updated, training_df)
            results.append(MSE)
        lambdaValue = lambdaValue * 10
    return results

def startCrossValidation(train_df, k, startLambda, stopLambda, runTestCasesGeneric):
    train_df = train_df.fillna(0)
    k_fold = []

    start = 0;
    foldLimit = 43 #Computed step size or it could be value of floor of train_df/10
    for i in range(k): #Since we have to run 10 fold Cross Validation
        if (start == 0):
            testing_df, training_df  = train_df[0:foldLimit], train_df[foldLimit:]
        else:
            testing_df, training_df = train_df[start:start + foldLimit], train_df.ix[0:start].append(train_df.ix[start+foldLimit:])

        start = start + foldLimit

        mean = list(training_df.mean())
        var = list(training_df.std())

        training_df = (training_df - mean)/var
        testing_df = (testing_df- mean)/var

        results = startRidgeRegression(training_df, testing_df, startLambda, stopLambda, runTestCasesGeneric, False)
        k_fold.append(results)
    return k_fold

def handleCV(k_fold):
    print "\n"
    print "%%%%%%%%%%%% CROSS VALIDATION ARRAY FOR VARIOUS VALUES OF LAMBDA %%%%%%%%%%"
    time.sleep(2)
    CV_line = []
    line = []
    for item in range(6):
        sum = 0;
        for i in range(10): #Since 10 fold
            sum = sum + float(k_fold[i][item])
        line.append(sum/float(10))
    CV_line.append(line)
    print pd.DataFrame(CV_line, columns=[0.0001, 0.001, 0.01, 0.1, 1, 10])

def displayMain(result):
    print pd.DataFrame(result, columns=[0.0001, 0.001, 0.01, 0.1, 1, 10])

""""
if __name__=="__main__":

    train_df = pd.read_table("trial1.txt",
                       delimiter=',',
                       header=None);

    test_df = pd.read_table("trial2.txt",
                       delimiter=',',
                       header=None);

    train_df = train_df.fillna(0)

    mean = list(train_df.mean()) #mean[-1] = 0   #var[-1] = 1
    var = list(train_df.std())

    training_df = (train_df - mean)/var
    testing_df = (train_df- mean)/var

    results = startRidgeRegression(training_df, testing_df, 0.01, 1.0)
    display(results)

    k_fold = startCrossValidation(train_df, 10, 0.0001, 10)
    print " ##########################################################################################################"
    print k_fold

    displayMain(k_fold)
    handleCV(k_fold)

    CV_line = []
    for item in range(6):
        sum = 0;
        for i in range(10): #Since 10 fold
            sum = sum + float(k_fold[i][item])
        CV_line.append(sum/float(10))
    #k_fold
    """




