from sklearn.model_selection import RepeatedKFold
from sklearn import svm, metrics
from scipy import stats
import pandas as pd
import numpy
import training

training.createdata()
dataset = pd.read_csv('dir/hog.csv')
dataset = dataset[(numpy.abs(stats.zscore(dataset)) < 5.04).all(axis=1)]
random_state = 12883823
rkf = RepeatedKFold(n_splits=5, n_repeats=30, random_state=random_state)
result = next(rkf.split(dataset), None)

data_train = dataset.iloc[result[0]]
data_test = dataset.iloc[result[1]]

data = data_train.iloc[:, [0, 3780]]
target = data_train.iloc[:, [3781]]

classifier = svm.SVC(C=1, gamma=0.1)
classifier.fit(data, target)

dataset_teste = pd.read_csv('dir/test_hog.csv')

predicted = classifier.predict(dataset_teste.iloc[:, [0, 3780]])
print(metrics.classification_report(dataset_teste.iloc[:, [3781]], predicted))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(dataset_teste.iloc[:, [3781]], predicted))
print(classifier.score(dataset_teste.iloc[:, [0, 3780]], dataset_teste.iloc[:, [3781]]))

