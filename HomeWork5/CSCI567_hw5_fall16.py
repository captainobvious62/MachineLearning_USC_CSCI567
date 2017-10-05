
import random
import sys
import matplotlib.pyplot as plt
from itertools import cycle
import KernelKMeans
cycol = cycle(['red', 'cyan', 'black', 'blue', 'magenta', 'green', '#FFC400', '#009E73', '#77BEDB', '#C4AD66']).next

def getNearestCluster(item, centers):

    nearest = sys.maxint
    nearest_dist = sys.maxint

    for i in range(0, len(centers)):
        center = centers[i]
        dist = (center[0] - item[0])**2 + (center[1] - item[1])**2
        if (dist < nearest_dist):
            nearest_dist = dist
            nearest = i
    return nearest

def getInitialRandomCenters(pointData, K):

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

def plotClusters(pointData, K):
    centers=getInitialRandomCenters(pointData, K)

    stillChanging=True

    while(stillChanging):
        #Create Empty cluster list
        splitData=[]
        for i in range(K):
            splitData.append([])

        assigned=False
        for item in pointData:
            nearest = getNearestCluster(item, centers)
            if(item[2] != nearest):
                item[2] = nearest #change the cluster number
                splitData[nearest].append(item)
                if (not assigned): #Not yet changed but this is a chnaged instance
                    stillChanging=True
                    assigned=True
            else:
                splitData[nearest].append(item)
        if(not assigned):
            stillChanging=False

        #New clusters
        if (stillChanging):
            centers=[]
            numCenter=0
            for cluster in splitData:
                xList=list(map(lambda x: float(x[0]), cluster))#find the sum of x coordinates in given cluster
                x = sum(xList)/len(xList)
                yList = list(map(lambda x: x[1], cluster))#find the sum of x coordinates in given cluster
                y = sum(yList)/len(yList)

                centers.append([x,y,numCenter])
                numCenter=numCenter + 1
        #print " STILL CHANGING = ", stillChanging
    return splitData

if __name__=="__main__":

    for K in [2, 3, 5]:
        for file in ["hw5_circle.csv", "hw5_blob.csv"]:
            print " Running for K = ", K, " and file = ", file
            pointData=[]
            for line in open(file):
                line = line.strip()
                line = line.split(",")
                line = [float(x) for x in line]
                line.append(-1)
                pointData.append(line)

            clusters=plotClusters(pointData, K)

            sumTotal=0
            for cluster in clusters:
                sumTotal = sumTotal + len(cluster)
            print " LENGTH = ", sumTotal

            setColors=set()
            for cluster in clusters:
                xList=list(map(lambda x: x[0], cluster))
                yList=list(map(lambda x: x[1], cluster))

                plt.scatter(xList, yList, c=cycol())
                #random.choice(['red', 'cyan', 'black', 'blue', 'magenta', 'green', '#FFC400', '#009E73', '#77BEDB', '#C4AD66']))
            plt.show()
    KernelKMeans.kkMeans()