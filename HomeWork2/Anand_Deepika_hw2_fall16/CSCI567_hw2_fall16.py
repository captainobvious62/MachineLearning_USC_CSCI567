import numpy as np
import pandas as pd
import copy
import time
import RidgeRegression
import PlotRoutine

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H

def findMax4MI(train_df):
    (m,n) = train_df.shape # n= 14
    target_col = train_df[n-1]
    MI_record = dict()
    bins = 10

    for i in range(0, n-1):
        source_col = train_df[i]
        c_XY = np.histogram2d(source_col,target_col,bins)[0]
        c_X = np.histogram(source_col,bins)[0]
        c_Y = np.histogram(target_col,bins)[0]

        H_X = shan_entropy(c_X)
        H_Y = shan_entropy(c_Y)
        H_XY = shan_entropy(c_XY)

        MI = H_X + H_Y - H_XY
        MI_record[i] = MI
    return MI_record

def selectFrame(df , columns):
    temp_df = df.ix[0:, columns]
    return getNormalizedDataFrame(temp_df)

def y_hat(x, w):
    return x.dot(w)

def runTestCases(W, df):

    df = df.fillna(0)

    numpyMatrix = df.as_matrix()
    sum = 0.0

    #print " testing will be done of ", numpyMatrix.shape[0], " samples"
    for i in range(numpyMatrix.shape[0]):
        row = np.array(numpyMatrix[i:i+1, 0:13])
        targetValue = numpyMatrix[i,13]

        one_column = np.ones(shape=(1, 1)) # Create a matrix with rows and 1 column
        row = np.c_[ one_column, row ]

        predictedValue = np.linalg.det(row.dot(W))
        sum = sum + (targetValue - predictedValue) ** 2

    return sum/ float(numpyMatrix.shape[0])


def runLinearRegression(df):
     numpyMatrix = df.as_matrix()

     x = np.array(numpyMatrix[:, 0:13]) # Select all rows all columns but the last column itself

     (m,n) = x.shape
     one_column = np.ones(shape=(m, 1)) # Create a matrix with m rows and 1 column

     x = np.c_[ one_column, x ] # now x conatins values of the form 1, x1, x2,...xn

     m,n=np.shape(x)

     y = np.array(numpyMatrix[:, 13:14]) # this is the target value
     w = np.array([np.ones(n)]).T

     w_updated = gradient_descent(x, y, w, 7)

     return w_updated

def runGenericLinearRegression(df):
    numpyMatrix = df.as_matrix()
    x = np.array(numpyMatrix[:, 0:numpyMatrix.shape[1]-1]) # Select all rows all columns but the last column itself

    (m,n) = x.shape
    one_column = np.ones(shape=(m, 1)) # Create a matrix with m rows and 1 column

    x = np.c_[ one_column, x ] # now x conatins values of the form 1, x1, x2,...xn
    m,n=np.shape(x)

    y = np.array(numpyMatrix[:, numpyMatrix.shape[1]-1:numpyMatrix.shape[1]]) # this is the target value
    w = np.array([np.ones(n)]).T
    w_updated = gradient_descent(x, y, w, 7)

    return w_updated

def gradient_descent(x, y, w, max_iter, alpha = 0.05):
    N = y.shape[0]

    for i in range(0, max_iter):
        XW = y_hat(x, w)

        gradient = np.dot(x.T, XW - y) / N

        w = w - alpha * gradient

    return w

def  runTestCasesGeneric(W, df):
    df = df.fillna(0)

    numpyMatrix = df.as_matrix()
    sum = 0.0
    residue = []

    columns = numpyMatrix.shape[1] - 1
    #print " testing will be done of ", numpyMatrix.shape[0], " samples"
    for i in range(numpyMatrix.shape[0]): #Run test case for every row
        row = np.array(numpyMatrix[i:i+1, 0:columns])
        targetValue = numpyMatrix[i,columns]

        one_column = np.ones(shape=(1, 1)) # Create a matrix with rows and 1 column
        row = np.c_[ one_column, row ]

        predictedValue = np.linalg.det(row.dot(W))
        sum = sum + (targetValue - predictedValue) ** 2
        residue.append(targetValue - predictedValue)
    #print "MSE = ", sum/ float(numpyMatrix.shape[0])
    return residue, sum/ float(numpyMatrix.shape[0])

def GetNewFeatureSet(train_df):

    df = train_df.ix[:, train_df.columns != 13] #Remove last coumn from df
    target_column = train_df.ix[:, 13:14]

    numpyMatrix = df.as_matrix()
    #print "Original size of matrix = ", numpyMatrix.shape
    count = 0
    for i in range(0, 13):
        for j in range( i, 13):
            col1 = numpyMatrix[:, i:i+1]
            col2 = numpyMatrix[:, j:j+1]

            col3 = col1 *col2
            numpyMatrix = np.append(numpyMatrix, col3, 1)

            count = count + 1
    numpyMatrix = np.append(numpyMatrix, target_column, 1)
    return numpyMatrix

def getNormalizedDataFrame(temp_df):
    mean = list(temp_df.mean())
    variance = list(temp_df.std())

    temp_df = (temp_df - mean)/variance
    return temp_df

def getKMax(correlation, K):
    corr_dict = dict()
    for i in range(len(correlation)):
        corr_dict[i] = abs(correlation[i])
    #print " Original dictionary = ", corr_dict
    corr_dict = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
    #print corr_dict
    return corr_dict[0:K]

def getDFUsingKMax(kMaxCorr, train_df):
    columns = [] # Add the target in df
    for item in kMaxCorr:
        columns.append(item[0]) #Add all indices of k maximum values
    columns.append(13)
    #print "Selected columns : ", columns
    df = train_df.ix[:, columns] #Only select maximum k columns and target column
    return df

def getCorrelation(saveDf, attributeUsed, residue):
    df = pd.DataFrame(data=residue)
    #print " len(residue)= " , len(residue), " size of Df = ", saveDf.shape, " df sie = ", df.shape
    #print " Used attributes : ", attributeUsed
    correlation = saveDf[saveDf.columns[0:-1]].apply(lambda x: x.corr(df[0])) # remove target column here itself
    correlation = correlation.drop(attributeUsed)
    correlation = correlation.abs()
    #print " Correlation found = ", correlation
    return correlation.idxmax()

def FormDFWithGivenIndex(saveDf, attributeUsed):
    temp_attrib = copy.copy(attributeUsed)
    temp_attrib.append(13) #Add target column
    return saveDf.ix[:, temp_attrib];

def featureSelectionRandomMI(train_df):
     count = 0
     frame_history = dict();
     print "....................Please wait ! ! Linear regression.................."
     print ".........It will pick 4 features from 13 randomly and report MSE............."
     print "........It might take till 20 seconds............."
     for i in range(0, 13):
         for j in range( i+1, 13):
             for k in range ( j+1, 13):
                 for l in range ( k+1, 13):
                     df = selectFrame(train_df, [i, j, k, l, 13]) # append target index
                     w = runGenericLinearRegression(df)
                     residue, MSE = runTestCasesGeneric(w, selectedFrame)
                     frame_history[str(i)  + "," + str(j) + "," + str(k) + "," + str(l)] = MSE
                     count = count + 1
                     if ( count == 100 or count == 200 or count == 400 or count == 600):
                         print ".....Please wait done with ......", count, " sets......"
                     #print " Running for count = " , count
     print ".............Completed Linear regression on ", count, " number of combinations............"
     frame_history = sorted(frame_history.items(), key=lambda x: x[1])
     print "Minimum MSE is = " , frame_history[0][1], " for feature set = ",  frame_history[0][0]

if __name__=="__main__":

     train_df = pd.read_table("trial1.txt",
                       delimiter=',',
                       header=None);

     test_df = pd.read_table("trial2.txt",
                       delimiter=',',
                       header=None);
     print "##########################  STARING 3.1  ##########################"

     print "~~~~~~~~~~~~~~~~~~~~~~~3.1 (a) Splitting the data in test data and train data~~~~~~~~~~~~~~~~~~~~~"
     test_df = test_df.fillna(0)
     train_df = train_df.fillna(0)
     print " Size of train data : ", train_df.shape
     print " Size of test data : ", test_df.shape

     print "\n"
     print "~~~~~~~~~~~~~~~~~~~~~~~3.1 (b) Histogram plot - Will be printed in the end~~~~~~~~~~~~~~~~~~~~~~~"
     time.sleep(2)
     print "\n"
     print "~~~~~~~~~~~~~~~~~~~~~~~3.1 (c) Correlation of all the columns with the target column~~~~~~~~~~~~~~~~~~~~~~~"
     correlation = train_df[train_df.columns[0:-1]].apply(lambda x: x.corr(train_df[13]))
     correlation = [round(item, 3) for item in correlation]
     print list(correlation)

     #Normalize training set
     mean = list(train_df.mean()) #mean[-1] = 0   #var[-1] = 1
     var = list(train_df.std())

     training_df = (train_df - mean)/var
     testing_df = (test_df- mean)/var

     print "\n"
     print "~~~~~~~~~~~~~~~~~~~~~~~3.1 (d) Normalizing data set complete~~~~~~~~~~~~~~~~~~~~~~~"
     # TODO: PLOT EACH COLUMN WILL COME HERE
     #for i in range(13):
     #   train_df[i].plot()
     #train_df[10].plot()
     #plt.figure();
     #plt.plot();

     print "\n"
     print "###################### STARTING 3.2 ######################"
     print "~~~~~~~~~~~~~~~~~~~~~~ 3.2 (a) Running Linear regression on train and test data~~~~~~~~~~~~~~~~~~~~~"
     w = runLinearRegression(training_df)
     print "MSE_(tested against Test data set)  " , runTestCases(w, testing_df)
     print "MSE_(tested against Train data set) " , runTestCases(w, training_df)

     print "\n"
     print "~~~~~~~~~~~~~~~~~~~~~~ 3.2 (b) Running Ridge regression on train and test data~~~~~~~~~~~~~~~~~~~~~"
     ridgeResult = RidgeRegression.startRidgeRegression(training_df, testing_df, 0.01, 1.0, runTestCasesGeneric, True)
     RidgeRegression.display(ridgeResult)

     print " Now will be running 10 fold Cross Validation with lambda ranging form 0.0001 to 1.0"
     time.sleep(2)

     k_fold = RidgeRegression.startCrossValidation(train_df, 10, 0.0001, 10, runTestCasesGeneric)
     print "%%%%%%%%%%%%%%%%%%%%%% Ridge Regression Results are now available for various lambdas over Cross validation data set %%%%%%%%%%%%%%%%%%%"

     time.sleep(2)
     RidgeRegression.displayMain(k_fold)
     RidgeRegression.handleCV(k_fold)
     print "\n"

     time.sleep(1)
     print "###################### 3.3 FEATURE SELECTION ######################"
     time.sleep(2)
     correlation = [abs(x) for x in correlation]
     kMaxCorr = getKMax(correlation, 4)
     kAttributeDF = getDFUsingKMax(kMaxCorr, train_df)
     kAttributeDF = getNormalizedDataFrame(kAttributeDF)

     w = runGenericLinearRegression(kAttributeDF)
     temp, MSE = runTestCasesGeneric(w, kAttributeDF)
     print "~~~~~~~~~~~~~~~~~~~~~~~ 3.3(a) Selecting 4 maximum correlated features ~~~~~~~~~~~~~~~~~~~~~~~"
     time.sleep(1)
     print " MSE_(Selected 4 max correlated indices)  " ,MSE

     #3.3 b part
     print "~~~~~~~~~~~~~~~~~~~~~~~3.3(b) Feature selection one at a time ~~~~~~~~~~~~~~~~~~~~~~~"
     time.sleep(1)

     attributeUsed = []
     saveDf = getNormalizedDataFrame(train_df)
     kMaxCorr = getKMax(correlation, 1)
     attributeUsed.append(kMaxCorr[0][0])
     kAttributeDF = getDFUsingKMax(kMaxCorr, train_df)
     kAttributeDF = getNormalizedDataFrame(kAttributeDF)
     w = runGenericLinearRegression(kAttributeDF)
     residue, MSE = runTestCasesGeneric(w, kAttributeDF)
     print "MSE after first run = ", MSE
     #print "After first run residue = ", residue

     maxCorrelatedIndex = getCorrelation(saveDf, attributeUsed, residue);
     #print "Max correlated index found = ", maxCorrelatedIndex
     attributeUsed.append(maxCorrelatedIndex)
     kAttributeDF = FormDFWithGivenIndex(saveDf, attributeUsed)
     w = runGenericLinearRegression(kAttributeDF)
     #print " For generic Linear regression w = ", w
     residue, MSE = runTestCasesGeneric(w, kAttributeDF)
     print "MSE after Second run = ", MSE
     #print "After second run residue = ", residue

     maxCorrelatedIndex = getCorrelation(saveDf, attributeUsed, residue);
     #print "Max correlated index found = ", maxCorrelatedIndex
     attributeUsed.append(maxCorrelatedIndex)
     kAttributeDF = FormDFWithGivenIndex(saveDf, attributeUsed)
     w = runGenericLinearRegression(kAttributeDF)
     #print " For generic Linear regression w = ", w
     residue, MSE = runTestCasesGeneric(w, kAttributeDF)
     print "MSE after Third run = ", MSE
     #print "After third run residue = ", residue

     maxCorrelatedIndex = getCorrelation(saveDf, attributeUsed, residue);
     #print "Max correlated index found = ", maxCorrelatedIndex
     attributeUsed.append(maxCorrelatedIndex)
     kAttributeDF = FormDFWithGivenIndex(saveDf, attributeUsed)
     w = runGenericLinearRegression(kAttributeDF)
     #print " For generic Linear regression w = ", w
     residue, MSE = runTestCasesGeneric(w, kAttributeDF)
     print "MSE after Fourth run = ", MSE

     mutual_information = findMax4MI(train_df)
     mutual_information = sorted(mutual_information.items(), key=lambda x: x[1], reverse=True)
     columns = []
     for k,v in  mutual_information[0:4]:
         columns.append(k)
     columns.append(13)

     print "~~~~~~~~~~~~~~~~~~~~~~~3.3(c) Feature selection of four columns with maximum mutual information ~~~~~~~~~~~~~~~~~~~~~~~"
     selectedFrame = selectFrame(train_df, columns)
     w = runGenericLinearRegression(selectedFrame)
     residue, MSE = runTestCasesGeneric(w, selectedFrame)
     print "MSE after selected 4 columns with maximum Mutual information = ", MSE

     print "~~~~~~~~~~~~~~~~~~~~~~~3.3(d) Feature selection of random four columns  mutual information ~~~~~~~~~~~~~~~~~~~~~~~"
     featureSelectionRandomMI(train_df)

     print "\n"
     print "###################### 3.4 POLYNOMIAL EXPANSION ######################"
     extraSetFeatureMatrix = GetNewFeatureSet(train_df)
     print "New Data set feature matrix dimensions : ", extraSetFeatureMatrix.shape
     temp_df = pd.DataFrame(data=extraSetFeatureMatrix)
     temp_df = getNormalizedDataFrame(temp_df)
     w = runLinearRegression(temp_df)
     print "MSE_(after polynomial expansion)" , runTestCases(w, temp_df)

     print "........Calling Graph routine to invoke plots........."
     time.sleep(3)
     PlotRoutine.plot_graphs(training_df)