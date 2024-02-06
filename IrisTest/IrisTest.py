
#import
from matplotlib import markers
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt # take little time...
import pandas as pd
import mglearn
from IPython.display import display

from sklearn.model_selection import train_test_split


#load test dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#check key of dataset
print("Keys of iris_dataset")
print(iris_dataset.keys())
#print(iris_dataset.feature_names) # feature name list ( num of feature )
#print(iris_dataset.target_names) # classification name
#print(iris_dataset.data) # feature data list (150, 4)
#print(iris_dataset.target) # classification data list (150,)
#print(type(iris_dataset.data))
#print(iris_dataset.data.shape) # show array-size of ndarray
#print(iris_dataset.data[:10, :2]) # use "slicing"
#print(iris_dataset.data_module)

# split dataset into "train-data" or "test-data"
X_train, X_test, y_train, y_test = train_test_split( iris_dataset.data, iris_dataset.target, random_state=111)
print("data shapes : ", X_train.shape, ":", X_test.shape, ":", y_train.shape, ":", y_test.shape)

# create pandas DataFrame
iris_dataFrame = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# print DataFrame
iris_table = iris_dataFrame.to_markdown(index=False)
##print(iris_table)

# create scatter matrix
grr = pd.plotting.scatter_matrix(iris_dataFrame, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)
##plt.show()

#********************************************************

# k-Nearest Neighbor estimation
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train, y_train)

# generate (1, 4) data, ( sklearn accept 2d-array )
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new) # estimated label number (as ndarray)
# X_new is (1, 4) array, so returned data is (1,) array.

print(iris_dataset.target_names[prediction])

#***************************************************
# evaluate test data

y_pred = knn.predict(X_test)
print(y_pred)

# check accuracy (use numpy.mean)
print("test data accuracy : {}".format(np.mean(y_pred == y_test)))

# check accuracy (use KNeighborClassifier.score)
print("test data accuracy(by knn) : {}".format(knn.score(X_test, y_test)))


print("Iris Test")