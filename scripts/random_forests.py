import pudb

from sklearn import tree
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn import cross_validation
import pandas as pd
import matplotlib.pyplot as plt
from time import time


print('reading the data')
X_train = pd.read_csv('../data/UCI_HAR_Dataset/train/X_train.txt', header=None, delim_whitespace=True)
y_train = pd.read_csv('../data/UCI_HAR_Dataset/train/y_train.txt', header=None, delim_whitespace=True)
X_test = pd.read_csv('../data/UCI_HAR_Dataset/test/X_test.txt', header=None, delim_whitespace=True)
y_test = pd.read_csv('../data/UCI_HAR_Dataset/test/y_test.txt', header=None, delim_whitespace=True) 

print('finished reading data')


clf = tree.DecisionTreeClassifier()

start = time()
clf = clf.fit(X_train, y_train)

print('Random forest took %.2f seconds' % ((time() - start)))
y_predict = clf.predict(X_test)


print(classification_report(y_test, y_predict))

print('------------------- Cross Validation ---------------')
pu.db
scores = cross_validation.cross_val_score(
        clf, X_train, y_train )

print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))  
