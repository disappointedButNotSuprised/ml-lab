# Decision Tree Classifier algorithm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.datasets import load_iris
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# load the dataset into parameter and target value vectors, then divide into train and test subsets
iris = datasets.load_iris()
X, y = datasets.load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create classifier
dtc = DecisionTreeClassifier(random_state=0, max_depth = 2)

# Fit the model and make a prediction
y_predict = dtc.fit(X_train, y_train).predict(X_test)

# generate metrics
conf_matrix = confusion_matrix(y_test, y_predict)
clf_report = classification_report(y_test, y_predict, target_names = iris.target_names, output_dict=True)

# print confusion matrix
plt.figure()
ax = sns.heatmap(conf_matrix, annot=True, fmt='d')
ax.set_title("Confusion matrix for DTC algorithm", pad=20, fontweight ="bold")
ax.set_xlabel("Predicted Diagnosis", fontsize=12, labelpad=10)
ax.xaxis.set_ticklabels(['Setosa', 'Virginica', 'Versicolor'])
ax.set_ylabel("Actual Diagnosis", fontsize=12, labelpad=10)
ax.yaxis.set_ticklabels(['Setosa', 'Virginica', 'Versicolor'])

# print classification report
plt.figure()
ax = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, )
ax.set_title("Classification report for DTC algorithm", pad=20, fontweight ="bold")

# hyperparameter controll test
metrics = list()
n_space = np.arange(2, 20, 1)
for min_samples in n_space:
    dtc_h = rfc = DecisionTreeClassifier(random_state=0, min_samples_split = min_samples)
    y_predict_h = dtc_h.fit(X_train, y_train).predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_predict_h)
    clf_report = classification_report(y_test, y_predict_h, target_names = iris.target_names, output_dict=True)
    prfs_report = precision_recall_fscore_support(y_test, y_predict_h, average='micro')
    metrics.append([prfs_report[0], prfs_report[1], prfs_report[2]])

# print matrics in relation to hyperparameter
metrics = np.array(metrics).T.tolist() 
N_space = np.linspace(n_space.min(), n_space.max(), 500)
plt.figure()
for i in range(len(metrics)):
    spline = make_interp_spline(n_space, metrics[i])
    plt.plot(N_space, spline(N_space), c = 'tab:blue')
plt.title('metrics vs min_samples_split', fontweight ="bold")
plt.xlabel('min_samples_split', fontsize=10)
plt.ylabel('ACC, TPR, F-score', fontsize=10)

plt.show()