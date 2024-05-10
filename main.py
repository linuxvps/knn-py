# Required imports
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


from DataSet import DatasetLoader
from SplitTrainAndTest import TrainAndTest
from naghasiKeshodan import naghashi


# Load the iris dataset
x, y = DatasetLoader().load_iris()
# Split the dataset into a training set and a test set
x_train, x_test, y_train, y_test = TrainAndTest().split_data(x, y, 0.2, 42)
# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, len(x_test))}

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

# Print the classification report
print(classification_report(y_test, y_pred))

# Output the best parameters and the best score
print(f"Best number of neighbors: {best_n_neighbors}")
print(f"Best score: {best_score}")

# Plotting the chart
mean_test_scores = knn_gscv.cv_results_['mean_test_score']
naghashi().plot_accuracy_vs_k_value(param_grid, mean_test_scores)

# Sort and print K values and scores from best to worst
k_scores = list(zip(param_grid['n_neighbors'], mean_test_scores))
sorted_k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)

print("K values sorted from best to worst:")
for k, score in sorted_k_scores:
    print(f"K={k}, Score={score:.4f}")
