from sklearn.metrics import classification_report


class print_data:
    def print_sorted_scores(self,param_grid, mean_test_scores,best_n_neighbors,best_score,y_test, y_pred):
        """
        Sorts and prints K values and scores from best to worst.

        :param param_grid: The dictionary of parameters to be tested.
        :param mean_test_scores: Mean test scores obtained during GridSearchCV.
        """
        # Print the classification report
        print(classification_report(y_test, y_pred))

        # Output the best parameters and the best score
        print(f"Best number of neighbors: {best_n_neighbors}")
        print(f"Best score: {best_score}")

        k_scores = list(zip(param_grid['n_neighbors'], mean_test_scores))
        sorted_k_scores = sorted(k_scores, key=lambda x: x[1], reverse=True)

        print("K values sorted from best to worst:")
        for k, score in k_scores:
            print(f"K={k}, Score={score:f}")
