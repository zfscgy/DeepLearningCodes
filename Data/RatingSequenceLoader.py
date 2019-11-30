import numpy as np


class RatingSeqDataLoader:
    def __init__(self, dataset, split=0.0, seq_len=8, label_mode="last", positive_rating=1):
        """

        """
        dataset_dict = {
            "m-100k": ("./Data/Datasets/MovieLens/Raw/ml-latest-small/ratings.csv", ","),
            'm-1m': ("./Data/Datasets/MovieLens/Raw/ml-1m/ratings.dat", "::"),
            'm-1m-u10i5': ("./Data/Datasets/MovieLens/Modified/filter_user_MovieLens-1M_After_2000-1-1", ","),
        }
        assert dataset in dataset_dict, "Invalid dataset"
        self.path = dataset_dict[dataset][0]
        self.delimiter = dataset_dict[dataset][1]
        self.user_rating_seqs = []
        self.n_items = 0
        self.n_users = 0
        self.train_test_split = split
        self.seq_len = seq_len
        self.label_mode = label_mode
        self._generate_rating_history_seqs(positive_rating, min_len=seq_len, train_test_split=split)

    def _generate_rating_history_seqs(self, positive_rating=1, split_time=3600000, min_len=8, train_test_split=0.9):
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
        user_rating_seqs = [seq for seq in user_rating_seqs if len(seq) > min_len + 2]
        np.random.shuffle(user_rating_seqs)
        print("Generated {0} sequences, total {1} items and {2} users".format(len(user_rating_seqs), n_items, n_users))
        self.user_rating_seqs = user_rating_seqs
        # plus 1 since the indices is start from 1 and 0 is remained for padding
        self.n_items = n_items + 1
        self.n_users = n_users + 1
        self.train_test_split = int(train_test_split * len(user_rating_seqs))

    def _get_rating_history_train_batch(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs[:self.train_test_split], batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - seq_len)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def _get_rating_history_test_batch(self, seq_len, batch_size):
        if batch_size is not None:
            seqs = np.random.choice(self.user_rating_seqs[self.train_test_split:], batch_size)
        else:
            seqs = self.user_rating_seqs[self.train_test_split:]
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - seq_len)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def _get_train_batch_from_all_user(self, seq_len, batch_size):
        seqs = np.random.choice(self.user_rating_seqs, batch_size)
        users = [seq[0] for seq in seqs]
        for i in range(len(seqs)):
            start_idx = np.random.randint(1, len(seqs[i]) - 1 - seq_len)
            seqs[i] = np.array(seqs[i][start_idx: start_idx + seq_len + 1])
        return np.array(list(seqs)), np.array(users)

    def _get_test_batch_from_all_user(self, seq_len, batch_size=None):
        if batch_size is not None:
            seqs = np.random.choice(self.user_rating_seqs, batch_size)
        else:
            seqs = self.user_rating_seqs
        users = [seq[0] for seq in seqs]
        seqs = [np.array(seq) for seq in seqs]
        for i in range(len(seqs)):
            seqs[i] = seqs[i][- seq_len - 1:]
        return np.array(list(seqs)), np.array(users)

    def get_train_batch(self, batch_size):
        if self.train_test_split == 0:
            xs, us = self._get_train_batch_from_all_user(self.seq_len, batch_size)
        else:
            xs, us = self._get_rating_history_train_batch(self.seq_len, batch_size)
        if self.label_mode == "last":
            return xs[:, :-1], xs[:, -1:]
        else:
            return xs[:, :-1], xs[:, 1:]

    def get_test_batch(self, batch_size=None):
        if self.train_test_split == 0:
            xs, us = self._get_test_batch_from_all_user(self.seq_len, batch_size)
        else:
            xs, us = self._get_rating_history_test_batch(self.seq_len, batch_size)
        if self.label_mode == "last":
            return xs[:, :-1], xs[:, -1:]
        else:
            return xs[:, :-1], xs[:, 1:]
