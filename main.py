# Required imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 31)}

# Use GridSearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=10)

# Fit the model to data
knn_gscv.fit(X_train, y_train)

# Check top performing n_neighbors value
best_n_neighbors = knn_gscv.best_params_['n_neighbors']

# Check mean score for the top performing value of n_neighbors
best_score = knn_gscv.best_score_

# Use the best parameters to make predictions
knn_best = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_best.fit(X_train, y_train)
y_pred = knn_best.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Output the best parameters and the best score
print(f"Best number of neighbors: {best_n_neighbors}")
print(f"Best score: {best_score}")

# Plotting the chart
mean_test_scores = knn_gscv.cv_results_['mean_test_score']
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, 31), mean_test_scores, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()

# Sort and print K values and scores from best to worst
k_values = list(range(1, 31))
k_scores = list(zip(k_values, mean_test_scores))
sorted_k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)

print("K values sorted from best to worst:")
for k, score in sorted_k_scores:
    print(f"K={k}, Score={score:.6f}")
