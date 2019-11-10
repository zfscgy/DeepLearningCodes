import numpy as np
from Data.Dataloader import RatingSeqDataLoader


class DataLoader(RatingSeqDataLoader):
    def __init__(self, dataset):
        dataset_dict = {
            "amazon": ("./Data/Datasets/TrashData/Rec/Amazon.csv", ","),
            "google": ("./Data/Datasets/TrashData/Rec/Google.csv", ","),
            "yoochoose": ("./Data/Datasets/TrashData/Rec/YooChoose.csv", ","),
            "lastfm": ("./Data/Datasets/TrashData/Rec/LastFM.csv", ",")
        }
        assert dataset in dataset_dict, "Invalid dataset"
        self.path = dataset_dict[dataset][0]
        self.delimiter = dataset_dict[dataset][1]
        self.user_rating_seqs = []
        self.n_items = 0
        self.n_users = 0
        self.train_test_split = 0

    def generate_rating_history_seqs(self, positive_rating=0, split_time=1e15, min_len=20, train_test_split=0.9):
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
                rating = float(rating)
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
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def get_rating_history_test_batch(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs[self.train_test_split:], batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - seq_len)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def get_train_batch_from_all_user(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs, batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - 1 - seq_len - 1)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def get_test_batch_from_all_user(self, seq_len, batch_size=None):
        if batch_size is not None:
            seqs = np.random.choice(self.user_rating_seqs[self.train_test_split:], batch_size)
        else:
            seqs = self.user_rating_seqs
        users = [seq[0] for seq in seqs]
        seqs = [np.array(seq) for seq in seqs]
        for i in range(len(seqs)):
            seqs[i] = seqs[i][- seq_len - 1:]
        return np.array(list(seqs)), np.array(users)
