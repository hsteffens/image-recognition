from sklearn import metrics
import pandas as pd
from sklearn.cluster import KMeans
import training

training.createdata()
dataset = pd.read_csv('dir/orb.csv')

dataset.fillna(0, inplace=True)
kmeans = KMeans(n_clusters=2, random_state=0).fit(dataset)
dataset_test = pd.read_csv('dir/test_orb.csv')

dataset_test.fillna(0, inplace = True)
predicted = kmeans.predict(dataset_test)

expected = list()
for i in range(32):
    expected.append(1)
for i in range(23):
    expected.append(0)

print(metrics.classification_report(expected, predicted))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))