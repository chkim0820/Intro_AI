# For CSDS 391 Project 2; classification using k-means clustering & linear classifier
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math


# Take the first n columns of n-dim data and convert them to vectors (numpy arrays)
def toVectors(data, dimension):
    vectorList = [] # For storing vectors for each dimension
    # Iterate through all rows in the data frame (all iris data)
    for i in range(len(data.index)):
        vector = np.array(data.iloc[i,:dimension]) # extract each vector (X)
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
def kMeansClustering(data, numClusters, dimension):
    D_values = []
    prevD = -1
    means = None
    # Loop or recursively run the method until D remains the same value for a few iterations
    while (True):
        # parse the data into a list of n-dimensional vectors
        vectors = toVectors(data, dimension)
        # calculate the updated mean for the kth cluster
        means, clusterMap = learningAlgo(vectors, means, numClusters, dimension)
        # calculate the objective function D; update belonging-ness to each cluster
        D = objectiveFunction(vectors, means, clusterMap)
        # Add current iteration's D to the list; to be plotted later
        D_values.append(D)
        # Check how D is changing
        if (D == prevD): # break the loop if D is the same; FIX to maybe less intense threshold
            break
        prevD = D # Saving current D for next iteration
    return D_values


def plotGraph(list, filename):
    plt.plot(list, '.-')
    plt.xlabel("Iteration")
    plt.ylabel("Value of Objective Function")
    plt.show()
    # plt.savefig(filename)


# Main method for running the current program
if __name__ == "__main__":
    data = pd.read_csv('irisdata.csv') # iris dataset to pd data frame; assuming same folder/directory

    # Exercise 1b; testing k-means clustering algorithm on irisdata.csv with k=2,3
    k2Values = kMeansClustering(data, 2, 4) # Learning algorithm on iris dataset; K=2 and 4 dimension
    plotGraph(k2Values, "Learning_Curve_K2")
    k3Values = kMeansClustering(data, 3, 4) # Learning algorithm on iris dataset; K=3 and 4 dimension
    plotGraph(k3Values, "Learning_Curve_K3")

    # Exercise 1c;