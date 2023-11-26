# For CSDS 391 Project 2 extra credit; neural network classifier for iris dataset
# Written by Chaehyeon Kim (cxk445)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Main method for running the current program
if __name__ == "__main__":
    data = pd.read_csv('irisdata.csv') # iris dataset to pd data frame; assuming same folder/directory
    features = data.iloc[:, :-1]  # Features (sepal/petal length/width)
    species = data.iloc[:, -1]   # Accurate classes; labels for species

    # Splitting between training vs. testing sets
    feat_train, feat_test, sp_train, sp_test = train_test_split(features, species, test_size=0.2, random_state=42)

    # Normalizing features
    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    # Creating a neural network using MLP classifier
    NN = MLPClassifier(hidden_layer_sizes=(10,), activation='logistic', max_iter=1000)

    # Training the model
    NN.fit(feat_train, sp_train)

    # Making predictions on the test set
    sp_pred = NN.predict(feat_test)

    # Evaluating the model's performance; results
    accuracy = accuracy_score(sp_test, sp_pred)
    confusion_mat = confusion_matrix(sp_test, sp_pred)
    class_report = classification_report(sp_test, sp_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification report:\n", class_report)