from sklearn.model_selection import train_test_split


class TrainAndTest:
    def __init__(self):
        self.x_train = None

    def split_data(self, x, y, test_size, random_state):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=test_size,
                                                                                random_state=random_state)
        return self.x_train, self.x_test, self.y_train, self.y_test
