### import the sklearn module for GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    return clf


def NBAccuracy(features_train, labels_train, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    
    ### create classifier
    # clf = GaussianNB()   
    ### fit the classifier on the training features and labels
    # clf.fit(features_train, labels_train)
    # OR
    clf = classify(features_train, labels_train)

    ### use the trained classifier to predict labels for the test features
    pred = clf.predict(features_test)

    ### calculate and return the accuracy on the test data
    ### this is slightly different than the example,
    ### where we just print the accuracy
    ### accuracy = no. of points classified coorectly / all points (in test set)
    ### method #1 : write code that compares predictions to y_test, elment-by-element
    Counts = 0
    for ii in range(len(features_test)):
        # if round(clf.predict([features_test[ii]])[0]) == labels_test[ii]:
        if pred[ii] == labels_test[ii]:
            Counts +=1
    accuracy_self_calculat = Counts / len(features_test)

    ### OR
    ### methos #2 : you might need to import an sklearn module
    accuracy = accuracy_score(pred, labels_test)
    print(accuracy, accuracy_self_calculat)
    return accuracy
