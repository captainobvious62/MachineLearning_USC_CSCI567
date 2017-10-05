
import random
import matplotlib.pyplot as plt
from itertools import cycle
import math
import numpy as np
import time
import sys
cycol = cycle(['red', 'cyan', 'black', 'blue', 'magenta', 'green', '#FFC400', '#009E73', '#77BEDB', '#C4AD66']).next

K=2
SIGMA=0.2
def readFile(filename):

    pointData=[]
    for line in open(filename):
        line = line.strip()
        line = line.split(",")
        line = [float(x) for x in line]
        line.append(-1)
        pointData.append(line)

    return pointData

def getInitialRandomCenters(pointData):

    centers = []
    numCenter = 0
    seenCenters = set()
    while numCenter < K:
        ptr = random.randrange(0, len(pointData), 1)
        if ptr not in seenCenters:
            pointData[ptr][2]=numCenter
            centers.append(pointData[ptr])
            numCenter = numCenter + 1
            seenCenters.add(ptr)
    return centers

# Item is the point for which we need to find  the cluster
# cluster is one entire cluster
def getDistance(cluster):
    N=len(cluster)
    distances=[ [ 0 for i in range(N) ] for j in range(N)]

    for i in range(N):
        for j in range(N):
            if (i==j):
                distances[i][j]=0
            elif(i<j):
                distances[i][j]=kernel(cluster[i], cluster[j])

    return sum(map(sum, distances))


def kernel(item, center):
    #return pow(2.718, -((center[0] - item[0])**2 + (center[1] - item[1])**2)/2*SIGMA*SIGMA)
    x = np.array(item)
    y = np.array(center)
    return math.sqrt(sum(x**2))*math.sqrt(sum(y**2))

def getMutualDistanceInCluster(pointData):
    cluster0=list(filter(lambda x: x[2] == 0, pointData))
    cluster1=list(filter(lambda x: x[2] == 1, pointData))

    d1=getDistance(cluster0)
    d2=getDistance(cluster1)

    return (d1, d2)

def getNearestDistance(item, pointData, d1, d2):
    cluster0=list(filter(lambda x: x[2] == 0, pointData))
    cluster1=list(filter(lambda x: x[2] == 1, pointData))

    dist1=0
    dist2=0

    for point in cluster0:
        dist1 = dist1 + kernel(item, point)
    effectiveDist1=((-2 * dist1)/len(cluster0)) + (d1/len(cluster0)**2)

    for point in cluster1:
        dist2 = dist2 + kernel(item, point)
    effectiveDist2=((-2 * dist2)/len(cluster1)) + (d2/len(cluster1)**2)

    return 0 if effectiveDist1 < effectiveDist2 else 1

def printClusters(pointData):
    print pointData
    (c0, c1)= (list(filter(lambda x: x[2] == 0, plotData)),list(filter(lambda x: x[2] == 1, plotData)))
    print " Cluster 0 size", len(c0)
    print " Cluster 1 size", len(c1)
    print pointData

def formClusters(pointData):
    #pointData=pointData[:20]

    getInitialRandomCenters(pointData)
    print " After initial init", pointData

    printClusters(pointData)
    stillChanging=True
    assigned=False

    iteration=0
    while(stillChanging and iteration < 5):

        for item in pointData:

            oldCluster=item[2]
            (d1, d2)=getMutualDistanceInCluster(pointData)
            newCluster=getNearestDistance(item, pointData, d1, d2)

            if (newCluster != oldCluster):
                item[2]=newCluster
                if (not assigned):
                    stillChanging=True
                    assigned=True

        if ( not assigned):
            stillChanging=False
        iteration=iteration+1
        print " At the end of iteration", iteration
        #printClusters(pointData)
        #drawClusters(plotData)
    return pointData

def drawClusters(pointData):
    (c0, c1)= (list(filter(lambda x: x[2] == 0, plotData)),list(filter(lambda x: x[2] == 1, plotData)))
    xList=list(map(lambda x: float(x[0]), c0))
    yList=list(map(lambda x: float(x[1]), c0))
    plt.scatter(xList, yList,c='cyan')

    xList=list(map(lambda x: float(x[0]), c1))
    yList=list(map(lambda x: float(x[1]), c1))
    plt.scatter(xList, yList)
    plt.scatter(xList, yList,c='red')
    plt.show()

def kkMeans():

    plotData=readFile("hw5_circle.csv")

    plotData=formClusters(plotData)
    (c0, c1)= (list(filter(lambda x: x[2] == 0, plotData)),list(filter(lambda x: x[2] == 1, plotData)))
    print " Cluster 0 size", c0
    print " Cluster 1 size", c1
    print " Final x"
    drawClusters(plotData)



