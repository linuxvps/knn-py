import matplotlib.pyplot as plt
import numpy as np


class naghashi:

    def plot_accuracy_vs_k_value(self,param_grid, mean_test_scores):
        """
        Plots the accuracy vs. K value.

        :param param_grid: The dictionary of parameters to be tested.
        :param mean_test_scores: Mean test scores obtained during GridSearchCV.
        """
        k_values = param_grid['n_neighbors']
        plt.figure(figsize=(12, 6))
        plt.plot(k_values, mean_test_scores, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Accuracy vs. K Value')
        plt.xlabel('K Value')
        plt.ylabel('Accuracy')
        plt.show()

