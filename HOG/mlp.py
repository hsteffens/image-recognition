from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
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

classifier = MLPClassifier(random_state=30, hidden_layer_sizes=8, learning_rate_init=0.1, momentum=0.9)
classifier.fit(data, target)

dataset_test = pd.read_csv('dir/test_hog.csv')

predicted = classifier.predict(dataset_test.iloc[:, [0, 3780]])
print(metrics.classification_report(dataset_test.iloc[:, [3781]], predicted))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(dataset_test.iloc[:, [3781]], predicted))
print(classifier.score(dataset_test.iloc[:, [0, 3780]], dataset_test.iloc[:, [3781]]))

