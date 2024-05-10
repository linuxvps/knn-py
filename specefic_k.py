from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
# Let's assume we want to use k=5 for this example
k = 80
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Step 4: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with k={}: {:.2f}%".format(k, accuracy * 100))

# Step 5: Make Predictions
# Let's say you have new test data X_new
X_new = [[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3], [7.7, 3.0, 6.1, 2.3]]
predicted_classes = model.predict(X_new)
print("Predicted classes for new data:")
for i in range(len(X_new)):
    print("Data {}: Predicted class {}".format(X_new[i], predicted_classes[i]))
