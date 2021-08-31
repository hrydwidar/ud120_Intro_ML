# !/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
from ClassifyNB import classify, Method_Accuracy
from class_vis import prettyPicture, output_image
import sys
from time import time

sys.path.append("../tools/")
from email_preprocess import preprocess



### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train = features_train[:round(len(features_train)/100)]
labels_train = labels_train[:round(len(labels_train)/100)]



#########################################################
### your code goes here ###
# Find the fitting data of features_train using Support Vector Machine (SVM) method
t0 = time()
clf = classify(features_train, labels_train, modle='SVM')
print("Training Time:", round(time()-t0, 3), "s")

# Predict the labels of features_test using SVM method
t0 = time()
Pred_labels_test = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Calculate the acuuracy of the method
t0 = time()
# Calculate the accuracy of the method
# NBAccuracy(features_train, labels_train, features_test, labels_test, modle='SVM')
Method_Accuracy(clf, features_test, labels_test)
print("Predicting Time:", round(time()-t0, 3), "s")

### draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
##############################################################
