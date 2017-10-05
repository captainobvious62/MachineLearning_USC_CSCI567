import random
import math
import numpy as np
import matplotlib.pyplot as plt

def getWeights(X, Y):
    # X will be received as it
    # Y will be in the transposed form that is always of the shape 10x1
    W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
    return W

def get1DWeights(X, Y):
    part1 = X.T.dot(X)
    part2 = X.T.dot(Y)
    return part2/part1

def adjustedWeights(W, lambdaVal):
    a = pow(W[0], 2)
    b = pow(W[1], 2)
    c = pow(W[2], 2)

    W[0] = W[0] + lambdaVal*(a + b + c)
    return W

def invokeHistogram(mse1, mse2, mse3, mse4, mse5, mse6):

    plt.hist(mse1)
    plt.title("g(x) = 1")
    plt.xlabel("Bins")
    plt.ylabel("Mean Squared Error")
    plt.show()

    plt.hist(mse2)
    plt.title("g(x) = w0")
    plt.xlabel("Bins")
    plt.ylabel("Mean Squared Error")
    plt.show()

    plt.hist(mse3)
    plt.title("g(x) = w0 + w1 * x")
    plt.xlabel("Bins")
    plt.ylabel("Mean Squared Error")
    plt.show()

    plt.hist(mse4)
    plt.title("g(x) = w0 + w1 * x + w2 * x2")
    plt.xlabel("Bins")
    plt.ylabel("Mean Squared Error")
    plt.show()

    plt.hist(mse5)
    plt.title("g(x) = w0 + w1 * x + w2 * x2 + w3 * x3")
    plt.xlabel("Bins")
    plt.ylabel("Mean Squared Error")
    plt.show()

    plt.hist(mse6)
    plt.title("g(x) = w0 + w1 * x + w2 * x2 + w3 * x3 + w4 * x4")
    plt.xlabel("Bins")
    plt.ylabel("Mean Squared Error")
    plt.show()

def runRegularizedRegression(N, M):
    rand_x = [ [ 0 for i in range(M)] for j in range(N)]
    rand_y = [ [ 0 for i in range(M)] for j in range(N)]


    for i in range(N):
        for j in range(M):
            rand_x[i][j] = round(random.uniform(-1, 1), 3)
            rand_y[i][j] = round(2 * rand_x[i][j] * rand_x[i][j] + random.gauss(0, math.sqrt(0.1)), 3)

    fy1 = np.ones((N,M))

    mse1 = np.zeros((N,), dtype=np.float);

    print " lambda     bias     variance"
    for lambdaVal in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        #print " .......Please wait. Running for lambda = ", lambdaVal
        for i in range(N):

            slice_x0 = np.ones((M,), dtype=np.float)
            slice_x1 = np.array(rand_x[i])
            slice_x2 = np.power(rand_x[i], 2)

            true_y = np.array(rand_y[i])

            X = np.vstack( (slice_x0, slice_x1, slice_x2) )
            #print " Shape of X = ", X.shape, " and X = ", X
            W = getWeights(X.T, true_y.T)
            W = adjustedWeights(W, lambdaVal)
            pred_y = W.dot(X)
            #print " Prediction : ", pred_y
            fy1[i] = pred_y
            #mse1[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));

        #bias and variance
        true_y = 2 * np.power(rand_x, 2)
        bias = np.mean(np.mean(np.power(np.subtract(true_y, fy1), 2)))
        #print " computed bias = ", bias

        meanVal = np.mean(fy1, axis = 1)
        #print " Means = ", meanVal
        for i in range(M):
            #print " Mean val selected = ", meanVal[i]
            fy1[i] = true_y[i] - meanVal[i]
            fy1[i] = np.power(fy1[i], 2)
        variance = np.mean(np.mean(fy1))

        print " {0:.3f}   {1:.3f}    {2:.3f}".format(lambdaVal, bias, variance)

def runRegression(N, M):

    rand_x = [ [ 0 for i in range(M)] for j in range(N)]
    rand_y = [ [ 0 for i in range(M)] for j in range(N)]


    for i in range(N):
        for j in range(M):
            rand_x[i][j] = round(random.uniform(-1, 1), 3)
            rand_y[i][j] = round(2 * rand_x[i][j] * rand_x[i][j] + random.gauss(0, math.sqrt(0.1)), 3)

    #print rand_x
    #print rand_y

    fy1 = np.ones((N,M))
    fy2 = np.zeros((N,M))
    fy3 = np.zeros((N,M))
    fy4 = np.zeros((N,M))
    fy5 = np.zeros((N,M))
    fy6 = np.zeros((N,M))

    mse1 = np.zeros((N,), dtype=np.float);
    mse2 = np.zeros((N,), dtype=np.float);
    mse3 = np.zeros((N,), dtype=np.float);
    mse4 = np.zeros((N,), dtype=np.float);
    mse5 = np.zeros((N,), dtype=np.float);
    mse6 = np.zeros((N,), dtype=np.float);

    for i in range(N): #Replace with N = 100

        slice_x0 = np.ones((M,), dtype=np.float)
        slice_x1 = np.array(rand_x[i])
        slice_x2 = np.power(rand_x[i], 2)
        slice_x3 = np.power(rand_x[i], 3)
        slice_x4 = np.power(rand_x[i], 4)

        true_y = np.array(rand_y[i])

        #For g(x) = 1
        pred_y = np.ones((M,), dtype=np.float) #all predictions will be equal to 1
        mse1[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));
        fy1[i] = pred_y

        #For g(x) = w0
        X = slice_x0 #All prediction will be equal to weight itself
        W = get1DWeights(X.T, true_y.T)
        pred_y = [ round(W, 3) for _ in range(M)]
        fy2[i] = pred_y
        mse2[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));

        #For g(x) = w0 + w1 * x
        X = np.vstack( (slice_x0, slice_x1) )
        W = getWeights(X.T, true_y.T)
        pred_y = W.dot(X)
        fy3[i] = pred_y
        mse3[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));

        #For g(x) = w0 + w1 * x + w2 * x2
        X = np.vstack( (slice_x0, slice_x1, slice_x2) )
        W = getWeights(X.T, true_y.T)
        pred_y = W.dot(X)
        fy4[i] = pred_y
        mse4[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));

        #For g(x) = w0 + w1 * x + w2 * x2 + w3 * x3
        X = np.vstack( (slice_x0, slice_x1, slice_x2, slice_x3) )
        W = getWeights(X.T, true_y.T)
        pred_y = W.dot(X)
        fy5[i] = pred_y
        mse5[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));

        #For g(x) = w0 + w1 * x + w2 * x2 + w3 * x3 + w4 * x4
        X = np.vstack( (slice_x0, slice_x1, slice_x2, slice_x3, slice_x4) )
        W = getWeights(X.T, true_y.T)
        pred_y = W.dot(X)
        fy6[i] = pred_y
        mse6[i] = np.mean(np.power(np.subtract(true_y, pred_y), 2));

    #Plot functions will go here

    #bias and variance
    true_y = 2 * np.power(rand_x, 2)
    bias2 = np.zeros((6,), dtype=np.float)
    bias2[0] = np.mean(np.mean(np.power(np.subtract(true_y, fy1), 2)))
    bias2[1] = np.mean(np.mean(np.power(np.subtract(true_y, fy2), 2)))
    bias2[2] = np.mean(np.mean(np.power(np.subtract(true_y, fy3), 2)))
    bias2[3] = np.mean(np.mean(np.power(np.subtract(true_y, fy4), 2)))
    bias2[4] = np.mean(np.mean(np.power(np.subtract(true_y, fy5), 2)))
    bias2[5] = np.mean(np.mean(np.power(np.subtract(true_y, fy6), 2)))

    #print bias2
    variance = np.zeros((6,), dtype=np.float)

    counter = 0;
    for fy in [fy1, fy2, fy3, fy4, fy5, fy6]:
        meanVal = np.mean(fy, axis = 1)

        for i in range(M):
            fy[i] = fy[i] - meanVal[i]
            fy[i] = np.power(fy[i], 2)
        variance[counter] = round(np.mean(np.mean(fy)), 3)
        counter = counter + 1

    #print variance

    print " Function     Bias2       Variance"
    for i in range(6):
        print " {0}        {1:.3f}     {2:.3f}".format("g" + str(i+1) + "(x)", bias2[i], variance[i])

    invokeHistogram(mse1, mse2, mse3, mse4, mse5, mse6)
if __name__ == "__main__":

    #runRegression(N=100, M=10)
    runRegression(N=100, M=100)

    #runRegularizedRegression(N=100, M=100)