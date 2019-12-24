import numpy as np
class MnistLoader:
    def __init__(self):
        train_paths = ["./Data/Datasets/MNIST/train-images.idx3-ubyte", "./Data/Datasets/MNIST/train-labels.idx1-ubyte"]
        test_paths = ["./Data/Datasets/MNIST/t10k-images.idx3-ubyte", "./Data/Datasets/MNIST/t10k-labels.idx1-ubyte"]
        train_images, train_labels = self.get_data_and_labels(train_paths[0], train_paths[1])
        test_images, test_labels = self.get_data_and_labels(test_paths[0], test_paths[1])
        self.train_set = np.concatenate([(np.array(train_images) - 128)/255, np.array(train_labels)[:, np.newaxis]], axis=1)\
            .astype(np.float32)
        self.test_set = np.concatenate([(np.array(test_images) - 128)/255, np.array(test_labels)[:, np.newaxis]], axis=1)\
            .astype(np.float32)

    @staticmethod
    def get_data_and_labels(images_filename, labels_filename):
        # print("Opening files ...")
        images_file = open(images_filename, "rb")
        labels_file = open(labels_filename, "rb")

        try:
            # print("Reading files ...")
            images_file.read(4)
            num_of_items = int.from_bytes(images_file.read(4), byteorder="big")
            num_of_rows = int.from_bytes(images_file.read(4), byteorder="big")
            num_of_colums = int.from_bytes(images_file.read(4), byteorder="big")
            labels_file.read(8)

            num_of_image_values = num_of_rows * num_of_colums
            data = [[None for x in range(num_of_image_values)]
                    for y in range(num_of_items)]
            labels = []
            for item in range(num_of_items):
                # print("Current image number: %7d" % item)
                for value in range(num_of_image_values):
                    data[item][value] = int.from_bytes(images_file.read(1),
                                                       byteorder="big")
                labels.append(int.from_bytes(labels_file.read(1), byteorder="big"))
            return data, labels
        except:
            print("Dataset not loaded")
        finally:
            images_file.close()
            labels_file.close()
            # print("Files closed.")

    def get_train_batch(self, batch_size):
        """
        return shape: [batch, 784], [batch]
        """
        idx = np.random.choice(self.train_set.shape[0], batch_size)
        train_batch = self.train_set[idx]
        return train_batch[:, :784], train_batch[:, -1].astype(np.int)

    def get_test_batch(self, batch_size=None):
        if batch_size is None:
            return self.test_set[:, :784], self.test_set[:, -1]
        idx = np.random.choice(self.test_set.shape[0], batch_size)
        test_batch = self.test_set[idx]
        return test_batch[:, :784], test_batch[:, -1].astype(np.int)