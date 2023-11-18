# For CSDS 391 Project 2; classification using k-means clustering & linear classifier
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
import numpy as np
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
    minD = math.inf # Smallest D occurred
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
            listMeans.append(means) # Append calculated means
            D_values.append(D) # Add current iteration's D to the list
            # Check how D is changing
            if (D == prevD): # break the loop if D is the same; FIX to maybe less intense threshold
                break
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


# For plotting D values
def plotD(list, filename):
    plt.plot(list, '.-')
    plt.xlabel("Iteration")
    plt.ylabel("Value of Objective Function")
    plt.show()
    # plt.savefig(filename)


# For plotting centers
def plotCenters(vectors, centers, map, filename):
    df = pd.DataFrame(np.row_stack(vectors))
    df.insert(len(df.columns), "Cluster", map)
    # Dataframe for each cluster
    cluster1 = df.query("Cluster==0", inplace=False)
    cluster2 = df.query("Cluster==1", inplace=False)
    cluster3 = df.query("Cluster==2", inplace=False)
    # Plot for each clusters
    plt.scatter(cluster1.loc[:,0], cluster1.loc[:,1])
    plt.scatter(cluster2.loc[:,0], cluster2.loc[:,1])
    plt.scatter(cluster3.loc[:,0], cluster3.loc[:,1])
    #Plot for centers
    # print(list(vector) for vector in centers[0])
    for center in centers:
        print(center)
        for vector in center:
            plt.scatter(vector[0], vector[1])
    # Set the axes names
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
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
    cen2Vectors, cen2Values, cen2Maps = kMeansClustering(data, 2, 2, retType="C", dimNames=dim)
    plotCenters(cen2Vectors, cen2Values, cen2Maps, "Petal_Center_K2")
    cen3Vectors, cen3Values, cen3Maps = kMeansClustering(data, 3, 2, retType="C", dimNames=dim)
    plotCenters(cen3Vectors, cen3Values, cen3Maps, "Petal_Center_K3")