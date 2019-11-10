import numpy as np

class TwoDDataGenerator:
    def __init__(self, train_size, test_size):
        self.train_index = train_size
        self.data = None

    def get_train_batch(self, batch_size):
        train_data = self.data[:self.train_index][np.random.choice(self.train_index, batch_size)]
        return train_data[:, :2], train_data[:, 2:]

    def get_test_data(self):
        return self.data[self.train_index:, :2], self.data[self.train_index:, 2:]

class MoonShapeDataGenerator(TwoDDataGenerator):
    def __init__(self, train_size=100, test_size=500):
        super().__init__(train_size, test_size)
        self.data = []
        for i in range(train_size + test_size):
            x, y = np.random.uniform(-1, 1, 2)
            while self.is_in_sample_range(x, y) == -1:
                x, y = np.random.uniform(-1, 1, 2)
            self.data.append([x, y, self.is_in_sample_range(x, y)])
        self.train_index = train_size
        self.data = np.array(self.data)

    def is_in_sample_range(self, x, y):
        if not (-1 <= x < 1 and -1 <= y < 1):
            return -1
        if x**2 + (y + 1)**2 <= 1 or y - 1 >= - 2 * x**2:
            return -1
        if y - 0.5 <= - 1.5 * x**2:
            return 0
        return 1

class AxeDataGenerator(TwoDDataGenerator):
    def __init__(self, train_size=100, test_size=500):
        super().__init__(train_size, test_size)
        self.data = []
        for i in range(train_size + test_size):
            x, y = np.random.uniform(-1, 1, 2)
            while self.is_in_sample_range(x, y) == -1:
                x, y = np.random.uniform(-1, 1, 2)
            self.data.append([x, y, self.is_in_sample_range(x, y)])
        self.train_index = train_size
        self.data = np.array(self.data)

    def is_in_sample_range(self, x, y):
        if not (-1 <= x < 1 and -1 <= y < 1):
            return -1
        if (x < -0.5 and abs(y) > 0.5) or (0 <= x and 0 < y < x and x**2 + y**2 < 1) or (y < 0 < x and 1 <= x**2 + y**2):
            return -1
        if (x < 0 and -0.5 <= y < 0) or (x > 0 and (-x <= y < 0 or 1 <= x**2 + y**2)):
            return 0
        return 1
