from sklearn.datasets import load_iris


class DatasetLoader:
    def __init__(self):
        self.data = None
        self.target = None

    def load_iris(self):
        iris = load_iris()
        return iris.data, iris.target
