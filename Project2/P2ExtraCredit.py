# For CSDS 391 Project 2 extra credit; neural network classifier for iris dataset
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# For running the neural network for classification
def neuralNetwork(features, species):
    # Splitting between training vs. testing sets
    feat_train, feat_test, sp_train, sp_test = train_test_split(features, species, test_size=0.25, random_state=42)

    # Normalizing features (input data)
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    # Creating a neural network using MLP classifier
    NN = MLPClassifier(hidden_layer_sizes=(5, 5), solver='lbfgs', max_iter=3000)
    NN.out_activation_ = 'softmax' # Applying softmax activation function

    # Training the model
    NN.fit(feat_train, sp_train)

    # Making predictions on the test set
    sp_pred = NN.predict(feat_test)

    return sp_test, sp_pred


# For printing out the results/analysis of the model's performance
def printNNResults(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    confusion_mat = confusion_matrix(actual, predicted)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)


# Main method for running the current program
if __name__ == "__main__":
    # For importing data set & dividing into two parts
    data = pd.read_csv('irisdata.csv') # iris dataset to pd data frame; assuming same folder/directory
    features = data.iloc[:, :-1]  # Features (sepal/petal length/width)
    species = data.iloc[:, -1]   # Accurate classes; labels for species

    # Building and running neural network
    sp_test, sp_pred = neuralNetwork(features, species)

    # Printing out the results
    printNNResults(sp_test, sp_pred)