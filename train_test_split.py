# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
# import sklear n libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# read the dataset.
iris_df = pd.read_csv("D:\\Git Github\\iris_project\\Iris.csv")
iris_df = iris_df.drop(columns=['Id'])
iris_df = iris_df.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width',
    'Species': 'species'
})

# convert species to numerical values
iris_df = iris_df.replace("Iris-setosa", 0)
iris_df = iris_df.replace("Iris-versicolor", 1)
iris_df = iris_df.replace("Iris-virginica", 2)

# define X and Y
X = iris_df.drop(columns=['species'])
Y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Accuracy with KNN
train_all = []
test_all = []
k_values = range(1, 40, 2)
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # train accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    train_all.append(train_acc)

    # test accuracy 
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_all.append(test_acc)

# plot the accuracies
# Plot
plt.figure(figsize=(8,5))
plt.plot(k_values, train_all, marker='o', label="Training Accuracy")
plt.plot(k_values, test_all, marker='s', label="Testing Accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs. k")
plt.legend()
plt.grid(True)
plt.show() 