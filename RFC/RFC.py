# rfc algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# load the dataset into parameter and target value vectors, then divide into train and test subsets
iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create classifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=2)

# Fit the model and make a prediction
y_predict = rfc.fit(X_train, y_train).predict(X_test)

# generate metrics
conf_matrix = confusion_matrix(y_test, y_predict)
clf_report = classification_report(y_test, y_predict, target_names = iris.target_names, output_dict=True)

# print confusion matrix
plt.figure()
ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
ax.set_title("Confusion matrix for RFC algorithm", pad=20, fontweight ="bold")
ax.set_xlabel("Predicted Diagnosis", fontsize=12, labelpad=10)
ax.xaxis.set_ticklabels(['Setosa', 'Virginica', 'Versicolor'])
ax.set_ylabel("Actual Diagnosis", fontsize=12, labelpad=10)
ax.yaxis.set_ticklabels(['Setosa', 'Virginica', 'Versicolor'])

# print classification report
plt.figure()
ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, )
ax.set_title("Classification report for RFC algorithm", pad=20, fontweight ="bold")

plt.show()