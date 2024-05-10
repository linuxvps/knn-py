import matplotlib.pyplot as plt
import numpy as np


class naghashi:

    def plot_accuracy_vs_k_value(self, param_grid, mean_test_scores, highlight_k=None, highlight_color='yellow'):
        """
        Plots the accuracy vs. K value.

        :param param_grid: The dictionary of parameters to be tested.
        :param mean_test_scores: Mean test scores obtained during GridSearchCV.
        :param highlight_k: The K value to highlight.
        :param highlight_color: The color to use for highlighting.
        """
        k_values = param_grid['n_neighbors']
        plt.figure(figsize=(12, 6))
        plt.plot(k_values, mean_test_scores, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)

        # Highlight the specified K value with the specified color
        if highlight_k is not None:
            highlight_index = np.where(k_values == highlight_k)[0]
            if len(highlight_index) > 0:
                plt.plot(highlight_k, mean_test_scores[highlight_index], marker='o', markersize=10,
                         color=highlight_color)

        plt.title('Accuracy vs. K Value')
        plt.xlabel('K Value')
        plt.ylabel('Accuracy')
        plt.show()
