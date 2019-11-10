from Data.Datasets.SimulatedData.TwoDSimData import AxeDataGenerator
import numpy as np
import matplotlib.pyplot as plt

moon_generator = AxeDataGenerator(300)
train_data = moon_generator.data[:moon_generator.train_index]
class_0_data = np.array([data[:2] for data in train_data if data[2] == 0])
class_1_data = np.array([data[:2] for data in train_data if data[2] == 1])
plt.plot(class_0_data[:, 0], class_0_data[:, 1], 'o')
plt.plot(class_1_data[:, 0], class_1_data[:, 1], 'x')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()