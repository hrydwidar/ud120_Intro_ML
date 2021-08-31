# !/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from ClassifyNB import classify, NBAccuracy


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
# Enter Your Code Here
# Find the fitting data of features_train using Naive Bayes method
t0 = time()
# clf = GaussianNB()
# clf.fit(features_train, labels_train)
clf = classify(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

# Predict the labels of features_test using Naive Bayes method
t0 = time()
Pred_labels_test = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Calculate the acuuracy of the method
t0 = time()
# ## use the trained classifier to predict labels for the test features
# pred = clf.predict(features_test)
# ## calculate and return the accuracy on the test data
# ## accuracy = no. of points classified coorectly / all points (in test set)
# ## method #1 : write code that compares predictions to y_test, elment-by-element
# Counts = 0
# for ii in range(len(features_test)):
#     if pred[ii] == labels_test[ii]:
#         Counts += 1
# accuracy_self_calculat = Counts / len(features_test)
# ## OR
# ## methos #2 : you might need to import an sklearn module
# accuracy = accuracy_score(pred, labels_test)
NBAccuracy(features_train, labels_train,  features_test, labels_test)
print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################
