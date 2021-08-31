### import the sklearn module for GaussianNB
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score


def classify(features_train, labels_train, modle='NB'):
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    
    ### your code goes here!
    
    if modle == 'NB':
        clf = naive_bayes.GaussianNB()
    # elif modle == 'SVM':
    else:
        """
        C :float, default = 1.0
            Regularization parameter. The strength of the regularization is inversely 
            proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.
        kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default =’rbf’
            Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, 
            ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If a callable is given it 
            is used to pre-compute the kernel matrix from data matrices; 
            that matrix should be an array of shape(n_samples, n_samples).
        degree: int, default = 3
            Degree of the polynomial kernel function(‘poly’). Ignored by all other kernels.
        gamma: {‘scale’, ‘auto’} or float, default =’scale’
            Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
            if gamma = 'scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma,
            if ‘auto’, uses 1 / n_features.
        """
        # SVC(kernel='linear', C=100, gamma=1000) # Accuracy = 99.43%
        # SVC(kernel='linear', C=1, gamma='scale')  # Accuracy = 98.4%
        clf = svm.SVC(kernel='linear')
    clf.fit(features_train, labels_train)  
    return clf


# def NBAccuracy(features_train, labels_train, features_test, labels_test, modle='NB'):
def Method_Accuracy(clf, features_test, labels_test):
    """ compute the accuracy of your Naive Bayes classifier """
    
    ### create classifier
    # clf = GaussianNB()   
    ### fit the classifier on the training features and labels
    # clf.fit(features_train, labels_train)
    # OR
    # clf = classify(features_train, labels_train, modle)

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
