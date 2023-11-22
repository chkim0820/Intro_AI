# For CSDS 391 Project 2; classification using k-means clustering & linear classifier
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
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
    

def boundaryFunction(x, a1, a2, b1, b2):
    if (a2 + b2 != 0):
        y = ((2*a1*x) - (2*a2*x) - a1**2 + a2**2 - b1**2 + b2**2)/(2 * (-b1 + b2))
    return y

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
            if (y1 + y2 != 0):
                y = boundaryFunction(x, x1, x2, y1, y2)
            else:
                exit("error")
            yValues.append(y)
        elif (numClusters == 3):
            y12 = boundaryFunction(x, x1, x2, y1, y2)
            y13 = boundaryFunction(x, x1, x3, y1, y3)
            y23 = boundaryFunction(x, x2, x3, y2, y3)
            for i in [y12, y13, y23]:
                yValues.append(i)
        xValues.append(x)
    return xValues, yValues


# For updating the weight
def updateWeight(weight, input, target, bias, N):
    epsilon = 0.1 / N
    if (target=='versicolor'):
        c = 0
    else:
        c = 1
    sig = sigmoid(np.dot(weight, input) + bias)
    newWeight = weight - (epsilon * (sig - c) * (sig * (1 - sig)) * input)
    return  newWeight


# For updating bias
def updateBias(oldBias, sum, target, N):
    epsilon = 0.1 / N
    if (target=='versicolor'):
        c = 0
    else:
        c = 1
    sig = sigmoid(sum + oldBias)
    newBias = oldBias - (epsilon * (sig - c) * (sig * (1 - sig)))
    return newBias


# For sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# # Neural network code that runs the data through the layer
def sigmoidNonlinearNN(data, patterns=None, iter=None):
    input = toVectors(data, 2, ['petal_length', 'petal_width'])
    N = len(input) # Total number of data
    weight = np.random.rand(2) # Uniform distribution
    weights = np.empty(N, dtype='object')
    predictions = [] # Saves the value after sigmoid; for identification purpose later
    badMSE = 0
    goodMSE = 0
    badWeight = None # Placeholder for first iteration's weight
    midWeight = None
    bias = 1
    midBias = 1
    badBias = 1
    mse = [] # List containing mean squared error of each iteration
    it = 0
    while (True if (iter==None) else it < iter + 1):
        predictions.clear()
        # Compute the weighted sum of inputs and bias & apply sigmoid function
        for i in range(N): # Dot product of weight and input data
            if (iter == None or (iter != None and it < iter)):
                species = data['species'].tolist()[i]
                bias = updateBias(bias, np.dot(input[i], weight), species, N)
                weight = updateWeight(weight, input[i], species, bias, N)
                weights[i] = weight
            weightedSum = np.dot(input[i], weight)
            predictions.append(sigmoid(weightedSum + bias))
        if (it == 0): # For saving "bad" results; earlier iterations
            badWeight = weight
            badBias = bias
        if (patterns != None):
            mse.append(meanSquaredError(patterns, predictions))
            if (it > 2):
                if (it == 3):
                    badMSE = mse[-1]
                if (mse[-1] < 0.05): # Stopping condition
                    goodMSE = mse[-1]
                    break
                if (mse[-1] > 0.3):
                    midWeight = weight
                    midBias = bias
        it += 1
    if (goodMSE == 0 and len(mse) > 1):
        goodMSE = mse[-1]
    return input, predictions, badMSE, goodMSE, bias, midBias, badBias, weight, midWeight, badWeight, mse


# # Calculate the mean-squared error
def meanSquaredError(patterns, predictions):
    mse = 0
    for i in range(len(patterns)):
        mse += np.square(predictions[i] - patterns[i])
    mse = mse / len(patterns)
    return mse


# Comparing good and bad weight samples
def compareMSE(weightedSums, badWeightedSums, patterns):
    goodMSE = meanSquaredError(patterns, weightedSums)
    badMSE = meanSquaredError(patterns, badWeightedSums)
    return goodMSE, badMSE


# For summed gradients
def summedGradient(data, bias, sums, patterns):
    grad1 = 0
    grad2 = 0
    N = len(data)
    for i in range(N):
        sig = sigmoid(sums[i] + bias)
        gradient = (sig - patterns[i]) * (sig * (1 - sig)) * data[i]
        grad1 += gradient[0] / N
        grad2 += gradient[1] / N
    gradients = [grad1, grad2]
    return gradients


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
            plt.plot(boundaryX, boundaryY[::3], color="c", label="Cluster 1 & 2")
            plt.plot(boundaryX, boundaryY[1::3], color="m", label="Cluster 1 & 3")
            plt.plot(boundaryX, boundaryY[2::3], color="y", label="Cluster 2 & 3")
    # Set the axes names
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()
    # Plot all vectors based on assigned clusters
    plt.show()
    # plt.savefig(filename)


# For plotting the NN decision boundary
def plotNNDecisionBoundary(data, result, weight=None, bias=None, simple=True):
    versicolorX = []
    versicolorY = []
    virginicaX = []
    virginicaY = []

    # Directing each point to appropriate classes based on NN result
    for i in range(len(result)):
        if (result[i] < 0.5):
            versicolorX.append(data[i][0])
            versicolorY.append(data[i][1])
        else:
            virginicaX.append(data[i][0])
            virginicaY.append(data[i][1])
    plt.scatter(versicolorX, versicolorY, label="versicolor")
    plt.scatter(virginicaX, virginicaY, color="tab:orange", label="virginica")

    # Determine the points along the boundary
    x = np.linspace(3, 7, 100) # x data limit
    if (simple==True):
        y = (-0.95 * x + 12.45) / 4.75
    else:
        y = (-weight[0] * x - bias)/(weight[1]) #generate corresponding point
        
    plt.plot(x, y, label='Decision Boundary', color='tab:gray') #plot the point
    # Set the axes names
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()
    # Plot all vectors based on assigned clusters
    plt.show()
    # plt.savefig(filename)        


# Plotting data of versicolor and virginica classes
def plotClasses(data):
    # Dataframe for each cluster
    versicolor = data.query("species=='versicolor'", inplace=False)
    virginica = data.query("species=='virginica'", inplace=False)
    # Plot for each clusters
    plt.scatter(versicolor['petal_length'], versicolor['petal_width'], color="tab:blue", label="versicolor")
    plt.scatter(virginica['petal_length'], virginica['petal_width'], color="tab:orange", label="virginica")
    # Set the axes names
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()

    # Plot all vectors based on assigned clusters
    plt.show()
    # plt.savefig(filename)


# For plotting 3D surface plots
def surfacePlot3D(data, outputs):
    fig = plt.figure()
    x = [vector[0] for vector in data]
    y = [vector[1] for vector in data]
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(x, y, np.array(outputs))
    
    # Label axes and title
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_zlabel('Neural Network Output')
    ax.set_title('Neural Network Output over Input Space')
    plt.show()


# Main method for running the current program
if __name__ == "__main__":
    data = pd.read_csv('irisdata.csv') # iris dataset to pd data frame; assuming same folder/directory

    # Exercise 1b; Test k-means clustering algorithm on irisdata.csv with k=2,3
    print("Plotting results of k-means clustering with k=2")
    k2Values = kMeansClustering(data, 2, 4, retType="D") # Learning algorithm on iris dataset; K=2 and 4 dimension
    plotD(k2Values, "Learning_Curve_K2")
    print("Plotting results of k-means clustering with k=3")
    k3Values = kMeansClustering(data, 3, 4, retType="D") # Learning algorithm on iris dataset; K=3 and 4 dimension
    plotD(k3Values, "Learning_Curve_K3")
    # Exercise 1c; Show the initial, intermediate, and converged cluster centers
    dim = ["petal_length", "petal_width"]
    cen2Vectors, cen2Means, cen2Maps = kMeansClustering(data, 2, 2, retType="C", dimNames=dim)
    print("Plotting converged cluster centers throughout for k=2")
    plotCenters(cen2Vectors, cen2Means, cen2Maps, 2, "Petal_Center_K2")
    cen3Vectors, cen3Means, cen3Maps = kMeansClustering(data, 3, 2, retType="C", dimNames=dim)
    print("Plotting converged cluster centers throughout for k=3")
    plotCenters(cen3Vectors, cen3Means, cen3Maps, 3, "Petal_Center_K3")
    # Exercise 1d; Plot decision boundaries; keeping 2 dimension
    xVal2, yVal2 = decisionBoundary(cen2Means[-1], 2)
    print("Plotting decision boundary for k=2")
    plotCenters(cen2Vectors, cen2Means, cen2Maps, 2, "Petal_Center_K2", xVal2, yVal2)
    xVal3, yVal3 = decisionBoundary(cen3Means[-1], 3)
    print("Plotting decision boundary for k=3")
    plotCenters(cen3Vectors, cen3Means, cen3Maps, 3, "Petal_Center_K3", xVal3, yVal3)


    # Exercise 2a; Plot the 2nd and 3rd iris classes
    print("Plotting classes for the 2nd and the 3rd classes")
    plotClasses(data)
    # Exercise 2b; Define a function that computes the output of simple one-layer neural network using a sigmoid non-linearity
    input = data.query("species=='versicolor' or species=='virginica'", inplace=False)
    patterns = []
    for species in input['species'].tolist():
        patterns.append(0 if (species=='versicolor') else 1)
    points, results, badMSE, goodMSE, bias, midBias, badBias, weight, midWeight, badWeight, mse = sigmoidNonlinearNN(input, iter=0)
    # Exercise 2c, e; Plots decision boundaries for non-linearity above
    print("Plotting results of the neural networks no iteration")
    plotNNDecisionBoundary(points, results, weight, bias)
    # Exercise 2d; 3D surface plot of output over the input space
    print("Printing 3D surface plot")
    surfacePlot3D(points, results)
    # Exercise 2e; show the results and compare the points
    # for i in range(len(results)):
    #     print(results[i], patterns[i])


    # Exercise 3a; mean-squared error calculation
    points, results, badMSE, goodMSE, bias, midBias, badBias, weight, midWeight, badWeight, mse = sigmoidNonlinearNN(input, patterns, iter=50)
    mse = meanSquaredError(patterns, results)
    # Exercise 3b; compare MSE for good and bad weight
    # goodMSE, badMSE = compareMSE(results, badMSE, patterns)
    print("Good and bad MSE", goodMSE, badMSE)
    # Exercise 3e; computes the summed gradient & plot how the decision boundary changes for a small step; FIX?
    gradients = summedGradient(points, bias, results, patterns)
    print("Plotting good weight")
    plotNNDecisionBoundary(points, results, weight, bias, False)
    print("Plotting bad weight; no gradient")
    plotNNDecisionBoundary(points, results, badWeight, badBias, False)


    # Exercise 4a; gradient descent
    points, results, badMSE, goodMSE, bias, midBias, badBias, weight, midWeight, badWeight, mse = sigmoidNonlinearNN(input, patterns)
    # Exercise 4b; decision boundary & learning curve
    print("Plotting for running gradients multiple times")
    plotNNDecisionBoundary(points, results, weight, bias, False)
    print("Plotting bad weight; no gradient")
    plotNNDecisionBoundary(points, results, badWeight, badBias, False)
    print("Plotting bad weight; no gradient")
    plotNNDecisionBoundary(points, results, midWeight, midBias, False)
    print("Plotting the learning curve for full gradient descents")
    plotD(mse, "NNLearningCurve")
    print("Printing 3D surface plot")
    surfacePlot3D(points, results)
    # For 2e data
    # for i in range(len(results)):
    #     print(results[i], patterns[i])