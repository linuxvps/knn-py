from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


call_count = 0  # تعریف متغیر برای شمارش تعداد فراخوانی‌ها


def calc_score_for_specific_k(k, X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy



def find_best_k(X_train, X_test, y_train, y_test, start, end):
    global call_count  # استفاده از متغیر جهانی برای شمارش فراخوانی‌ها
    call_count += 1  # افزایش شمارنده هر بار صدا زده شدن متد

    if start >= end:
        return start

    middle = (start + end) // 2

    left_accuracy = calc_score_for_specific_k((start + middle) // 2, X_train, X_test, y_train, y_test)
    middle_accuracy = calc_score_for_specific_k(middle, X_train, X_test, y_train, y_test)
    right_accuracy = calc_score_for_specific_k((middle + end) // 2, X_train, X_test, y_train, y_test)

    if middle_accuracy >= left_accuracy and middle_accuracy >= right_accuracy:
        return middle

    if right_accuracy > left_accuracy:
        return find_best_k(X_train, X_test, y_train, y_test, (middle + end) // 2, end)
    else:
        return find_best_k(X_train, X_test, y_train, y_test, start, (start + middle) // 2)


iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_k = find_best_k(X_train, X_test, y_train, y_test, 1, len(X_train))
specific_k = calc_score_for_specific_k(best_k, X_train, X_test, y_train, y_test)

print("Best K:", best_k)
print("Accuracy:", specific_k)
print("Number of calls to find_best_k:", call_count)  # چاپ تعداد فراخوانی‌ها در پایان برنامه
