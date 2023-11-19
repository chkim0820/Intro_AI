# For CSDS 391 Project 2; classification using k-means clustering & linear classifier
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import random
import math


# Take the first n columns of n-dim data and convert them to vectors (numpy arrays)
def toVectors(data, dimension, dimNames):
    vectorList = [] # For storing vectors
    if (dimNames == None): # If names of dimension columns not specified, choose first n dim.
        dimNames = data.columns[:dimension]
    # Iterate through all rows in the data frame (all iris data)
    for i in data.index:
        vector = np.empty(dimension)
        j = 0 # For indexing in vector
        for dim in dimNames:
            vector[j] = data.loc[i, dim] # extract each vector (X)
            j += 1
        vectorList.append(vector)
    return vectorList


# Implementation of the learning algorithm for new cluster means
def learningAlgo(vectors, means, numClusters, dimension):
    if (means == None): # initial means of each cluster; choose random, non-dupl vectors
        means = random.sample(vectors, numClusters)
    clusterMap = [] # Maps vectors to clusters
    totalSum = [np.zeros(dimension)] * numClusters # Contains the total sum of vectors in each cluster
    totalNum = [0] * numClusters # Contains the total num of vectors in each cluster

    # Assign each vectors to the clusters
    for n in range(len(vectors)):
        cluster = -1 # which cluster the vector belongs in 
        minNorm = math.inf # the minimal value ∥xn − µk∥^2 out of all possible k
        # Find which cluster the current vector belongs in 
        for k in range(numClusters):
            diff = np.subtract(vectors[n], means[k]) # xn − µk
            norm = np.linalg.norm(diff) # ∥xn − µk∥^2
            if (norm < minNorm):
                cluster = k
                minNorm = norm
        # Update info for calculating the new mean for each cluster
        clusterMap.append(cluster) # Maps current vector to 'cluster'
        totalSum[cluster] = np.add(totalSum[cluster], vectors[n]) 
        totalNum[cluster] = totalNum[cluster] + 1
    # Calculate the new mean for each clusters
    for k in range(numClusters):
        if (totalNum[k] > 0):
            means[k] = np.divide(totalSum[k], totalNum[k])
        else:
            means[k] = None # If cluster empty, take it out
    return means, clusterMap


# For calculating the new values to the objective function
def objectiveFunction(vectors, means, clusterMap):
    D = 0
    for n in range(len(vectors)):
        cluster = clusterMap[n] # Index of which cluster current vector belongs in
        mean = means[cluster] # Mean of the cluster determined above
        diff = np.subtract(vectors[n], mean) # xn − µk
        D = D + np.linalg.norm(diff) # Add current ∥xn − µk∥^2
    return D


# K-means clustering algorithm to be applied to the irisdata.csv dataset
def kMeansClustering(data, numClusters, dimension, retType, dimNames=None):
    D_values = [] # A list containing the D values for each iteration
    prevD = -1 # Stores the D value from the previous iteration
    means = None # Mean vectors for each cluster; updated
    clusterMap = None # Maps each value to appropriate clusters
    listMeans = [] # List of means/centers of clusters for each iteration
    vectors = toVectors(data, dimension, dimNames) # Parsing data into n-dimensional vectors
    firstIt = True

    # Loop or recursively run the method until D remains the same value for a few iterations
    for i in range(100): 
        # calculate the updated mean for the kth cluster
        means, clusterMap = learningAlgo(vectors, means, numClusters, dimension)
        # calculate the objective function D; update belonging-ness to each cluster
        D = objectiveFunction(vectors, means, clusterMap)
        if (D > prevD and not firstIt):
            means = listMeans[-1] # Try again
        else:
            # Check how D is changing
            if (D == prevD): # break the loop if D is the same; FIX to maybe less intense threshold
                break
            listMeans.append(means.copy()) # Append the calculated means
            D_values.append(D) # Add current iteration's D to the list
            prevD = D # Saving current D for next iteration
        firstIt = False
    # Return appropriate lists
    if (retType=="D"):
        return D_values
    elif (retType=="C"):
        centers = []
        lenMeans = len(listMeans) - 1
        centers.append(listMeans[0])
        centers.append(listMeans[int(lenMeans/2)])
        centers.append(listMeans[lenMeans])
        return vectors, centers, clusterMap
    

# Calculate the decision boundary based on the calculated means
def decisionBoundary(means, numClusters):
    xValues = []
    yValues = []
    x1, y1 = means[0] # 1st cluster
    x2, y2 = means[1] # 2nd cluster
    if (numClusters == 3):
        x3, y3 = means[2] # 3rd cluster
    #Find a line where the distance from both centers are equal for all x
    for x in range(8):
        if (numClusters == 2):
            y =  y1 + ((x - x1) ** 2 - (x - x2) ** 2 + (y2 ** 2 - y1 ** 2)) / (2 * (y2 - y1))
            # equation = sp.Eq((x1 - x)**2 + (y1 - y)**2, (x2 - x)**2 + (y2 - y)**2)
            yValues.append(y)
        elif (numClusters == 3):
            y12 = y1 + ((x - x1) ** 2 - (x - x2) ** 2 + (y2 ** 2 - y1 ** 2)) / (2 * (y2 - y1))
            y13 = y1 + ((x - x1) ** 2 - (x - x3) ** 2 + (y3 ** 2 - y1 ** 2)) / (2 * (y3 - y1))
            y23 = y2 + ((x - x2) ** 2 - (x - x3) ** 2 + (y3 ** 2 - y2 ** 2)) / (2 * (y3 - y2))
            for i in [y12, y13, y23]:
                yValues.append(i)
        xValues.append(x)
    return xValues, yValues


# For plotting D values
def plotD(list, filename):
    plt.plot(list, '.-')
    plt.xlabel("Iteration")
    plt.ylabel("Value of Objective Function")
    plt.show()
    # plt.savefig(filename)


# For plotting centers
def plotCenters(vectors, centers, map, numClusters, filename, boundaryX=None, boundaryY=None):
    df = pd.DataFrame(np.row_stack(vectors))
    df.insert(len(df.columns), "Cluster", map)
    # Dataframe for each cluster
    cluster1 = df.query("Cluster==0", inplace=False)
    cluster2 = df.query("Cluster==1", inplace=False)
    cluster3 = df.query("Cluster==2", inplace=False)
    # Plot for each clusters
    plt.scatter(cluster1.loc[:,0], cluster1.loc[:,1], color="tab:blue")
    plt.scatter(cluster2.loc[:,0], cluster2.loc[:,1], color="tab:orange")
    plt.scatter(cluster3.loc[:,0], cluster3.loc[:,1], color="tab:green")
    #Plot for centers
    xValues = []
    yValues = []
    colors = ["thistle", "darkviolet", "indigo"]
    for center in centers:
        for vector in center:
            xValues.append(vector[0])
            yValues.append(vector[1])
    plt.scatter(xValues, yValues, c=np.repeat(colors, numClusters))
    # Name the center plots
    inc = 2
    for dim in range(numClusters):
        plt.text(xValues[dim], yValues[dim], "1")
        plt.text(xValues[dim + inc], yValues[dim + inc], "2")
        plt.text(xValues[dim + (inc * 2)], yValues[dim + (inc * 2)], "3")
    # Plot decision boundaries if they exist
    if (boundaryX != None and boundaryY != None):
        if (numClusters == 2):
            plt.plot(boundaryX, boundaryY, color="tab:gray", label="Decision Boundary")
        elif (numClusters == 3):
            plt.plot(boundaryX, boundaryY[::3], color="tab:gray", label="1 & 2")
            plt.plot(boundaryX, boundaryY[1::3], color="tab:gray", label="1 & 3")
            plt.plot(boundaryX, boundaryY[2::3], color="tab:gray", label="2 & 3")
    # Set the axes names
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    # plt.legend()
    # Plot all vectors based on assigned clusters
    plt.show()
    # plt.savefig(filename)


# Main method for running the current program
if __name__ == "__main__":
    data = pd.read_csv('irisdata.csv') # iris dataset to pd data frame; assuming same folder/directory

    # # Exercise 1b; testing k-means clustering algorithm on irisdata.csv with k=2,3
    # k2Values = kMeansClustering(data, 2, 4, retType="D") # Learning algorithm on iris dataset; K=2 and 4 dimension
    # plotD(k2Values, "Learning_Curve_K2")
    # k3Values = kMeansClustering(data, 3, 4, retType="D") # Learning algorithm on iris dataset; K=3 and 4 dimension
    # plotD(k3Values, "Learning_Curve_K3")

    # Exercise 1c; show the initial, intermediate, and converged cluster centers
    dim = ["petal_length", "petal_width"]
    cen2Vectors, cen2Means, cen2Maps = kMeansClustering(data, 2, 2, retType="C", dimNames=dim)
    # plotCenters(cen2Vectors, cen2Means, cen2Maps, 2, "Petal_Center_K2")
    cen3Vectors, cen3Means, cen3Maps = kMeansClustering(data, 3, 2, retType="C", dimNames=dim)
    # plotCenters(cen3Vectors, cen3Means, cen3Maps, 3, "Petal_Center_K3")

    # Exercise 1d; Plot decision boundaries; keeping 2 dimension
    xVal2, yVal2 = decisionBoundary(cen2Means[-1], 2)
    plotCenters(cen2Vectors, cen2Means, cen2Maps, 2, "Petal_Center_K2", xVal2, yVal2)
    xVal3, yVal3 = decisionBoundary(cen3Means[-1], 3)
    plotCenters(cen3Vectors, cen3Means, cen3Maps, 3, "Petal_Center_K3", xVal3, yVal3)