# Required imports
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from DataSet import DatasetLoader
from SplitTrainAndTest import TrainAndTest
from naghasiKeshodan import naghashi
from print_k_value import print_data

# Load the iris dataset
x, y = DatasetLoader().load_iris()
# Split the dataset into a training set and a test set
x_train, x_test, y_train, y_test = TrainAndTest().split_data(x, y, 0.4, 42)
# Initialize the KNN classifier
knn = KNeighborsClassifier()
# Create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 80)}

# Use GridSearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=10)

# Fit the model to data
knn_gscv.fit(x_train, y_train)

# Check top performing n_neighbors value
best_n_neighbors = knn_gscv.best_params_['n_neighbors']

# Check mean score for the top performing value of n_neighbors
best_score = knn_gscv.best_score_

# Use the best parameters to make predictions
knn_best = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn_best.fit(x_train, y_train)
y_pred = knn_best.predict(x_test)

mean_test_scores = knn_gscv.cv_results_['mean_test_score']
# Print sorted K values and scores from best to worst
print_data().print_sorted_scores(param_grid, mean_test_scores, best_n_neighbors, best_score, y_test, y_pred)
# Plotting the chart
naghashi().plot_accuracy_vs_k_value(param_grid, mean_test_scores, best_n_neighbors, "green")
