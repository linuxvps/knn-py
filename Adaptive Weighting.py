# وزندهی تطبیقی (Adaptive Weighting)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import logging

# تنظیمات لاگ‌گذاری
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

# بارگذاری داده‌ها
data = load_iris()
X, y = data.data, data.target

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تابع محاسبه فاصله اقلیدسی
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# آموزش مدل شبکه عصبی برای وزن‌دهی تطبیقی و تعیین K
def train_weighting_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000)
    distances = []
    weights = []
    for i, x in enumerate(X_train):
        for j, x2 in enumerate(X_train):
            if i != j:
                distance = euclidean_distance(x, x2)
                distances.append(distance)
                weights.append(1 if y_train[i] == y_train[j] else 0)
    distances = np.array(distances).reshape(-1, 1)
    weights = np.array(weights)
    model.fit(distances, weights)
    return model

# ایجاد و آموزش مدل وزن‌دهی
weighting_model = train_weighting_model(X_train, y_train)

# تابع KNN با وزن‌دهی تطبیقی و تعیین خودکار K
def adaptive_weighted_knn(X_train, y_train, X_test, weighting_model):
    y_pred = []
    for x_test in X_test:
        distances = []
        for x_train in X_train:
            distance = euclidean_distance(x_test, x_train)
            distances.append(distance)

        distances = np.array(distances).reshape(-1, 1)
        weights = weighting_model.predict(distances)

        # تعیین K بهینه بر اساس وزن‌ها
        sorted_indices = np.argsort(weights)[::-1]
        cumulative_weights = np.cumsum(weights[sorted_indices])
        K = np.argmax(cumulative_weights >= cumulative_weights[-1] * 0.9) + 1

        neighbors = sorted_indices[:K]
        neighbor_weights = weights[neighbors]
        neighbor_labels = y_train[neighbors]

        weighted_votes = {}
        for label, weight in zip(neighbor_labels, neighbor_weights):
            if label in weighted_votes:
                weighted_votes[label] += weight
            else:
                weighted_votes[label] = weight

        predicted_label = max(weighted_votes, key=weighted_votes.get)
        y_pred.append(predicted_label)

        # لاگ‌گذاری مقدار K و جزئیات
        logger.info(f"نمونه تست: {x_test}")
        logger.info(f"فاصله‌ها: {distances.flatten()}")
        logger.info(f"وزن‌ها: {weights}")
        logger.info(f"اندیس‌های مرتب‌شده: {sorted_indices}")
        logger.info(f"وزن‌های تجمعی: {cumulative_weights}")
        logger.info(f"K انتخاب‌شده: {K}")
        logger.info(f"وزن همسایگان: {neighbor_weights}")
        logger.info(f"برچسب همسایگان: {neighbor_labels}")
        logger.info(f"برچسب پیش‌بینی‌شده: {predicted_label}")
        logger.info("-" * 40)

    return np.array(y_pred)

# پیش‌بینی با استفاده از KNN با وزن‌دهی تطبیقی و تعیین خودکار K
y_pred = adaptive_weighted_knn(X_train, y_train, X_test, weighting_model)

# محاسبه دقت
accuracy = accuracy_score(y_test, y_pred)
print(f"دقت مدل KNN با وزن‌دهی تطبیقی و تعیین خودکار K: {accuracy:.2f}")
