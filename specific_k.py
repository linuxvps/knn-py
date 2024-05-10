from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def calc_score_for_specific_k(k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def find_best_k(X_train, X_test, y_train, y_test):
    max_accuracy = 0
    best_k = 1

    for k in range(1, len(X_train)):
        accuracy = calc_score_for_specific_k(k)
        print(f"for k : {k} the score is {accuracy}")
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_k = k

    return best_k, max_accuracy


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_k, specific_k = find_best_k(X_train, X_test, y_train, y_test)
print("Best K:", best_k)
print("Accuracy:", specific_k)
