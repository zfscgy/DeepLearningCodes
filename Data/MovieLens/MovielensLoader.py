import numpy as np
class DataLoader:
    def __init__(self, dataset):
        dataset_dict = {
            "100k": ("Data/MovieLens/Raw/ml-latest-small/ratings.csv", ","),
            '1m': ("Data/MovieLens/Raw/ml-1m/ratings.dat", "::")
        }
        assert dataset in dataset_dict, "Invalid dataset"
        self.path = dataset_dict[dataset][0]
        self.delimiter = dataset_dict[dataset][1]
        self.user_rating_seqs = []
        self.n_items = 0
        self.n_users = 0
        self.train_test_split = 0

    def generate_rating_history_seqs(self, positive_rating=3, split_time=3600000, min_len=20, train_test_split=0.9):
        user_rating_seqs = []
        n_items = 0
        n_users = 0
        with open(self.path, "r") as ratings:
            line = ratings.readline()
            current_seq = []
            last_timestamp = None
            while line != "":
                user, item, rating, timestamp = line[:-1].split(self.delimiter)
                # Convert to integers
                user = int(user)
                if user > n_users:
                    n_users = user
                item = int(item)
                if item > n_items:
                    n_items = item
                rating = int(rating)
                timestamp = int(timestamp)

                if rating < positive_rating:
                    line = ratings.readline()
                    continue
                if len(current_seq) == 0:
                    current_seq.append(user)
                    current_seq.append(item)
                    last_timestamp = timestamp
                else:
                    if user == current_seq[0] and timestamp - last_timestamp < split_time:
                        current_seq.append(item)
                        last_timestamp = timestamp
                    else:
                        user_rating_seqs.append(current_seq)
                        current_seq = [user, item]
                line = ratings.readline()
            user_rating_seqs.append(current_seq)
        user_rating_seqs = [seq for seq in user_rating_seqs if len(seq) > min_len + 1]
        np.random.shuffle(user_rating_seqs)
        print("Generated {0} sequences, total {1} items and {2} users".format(len(user_rating_seqs), n_items, n_users))
        self.user_rating_seqs = user_rating_seqs
        # plus one since the indices is start from 1
        self.n_items = n_items + 1
        self.n_users = n_users + 1
        self.train_test_split = int(train_test_split * len(user_rating_seqs))

    def get_rating_history_train_batch(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs[:self.train_test_split], batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - 1 - seq_len)
            seqs[i] = seqs[i][start_idx: start_idx + seq_len + 1]
        return np.array(seqs), np.array(users)

    def get_rating_history_test_batch(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs[self.train_test_split:], batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - seq_len)
            seqs[i] = seqs[i][start_idx: start_idx + seq_len + 1]
        return np.array(list(seqs)), np.array(users)
