class RatingSeqDataLoader:
    """
    This class is a base Dataloder for rating sequence data
    """
    def generate_rating_history_seqs(self, positive_rating=0, split_time=1e15, min_len=20, train_test_split=0.9):
        """
        :param positive_rating: ratings below positive rating will be throw out
        :param split_time: Same user's contiguous two ratings, if time between them is longer than split time,
            then start a new sequence
        :param min_len: Sequence shorter than min_len will be dropped
        :param train_test_split:
        :return:
        """
        raise NotImplementedError()


    def get_rating_history_train_batch(self, seq_len, batch_size):
        """
        Randomly select sequences from training data
        :param seq_len:
        :param batch_size:
        :return:
        """
        raise NotImplementedError()

    def get_rating_history_test_batch(self, seq_len, batch_size):
        """
        Randomly select sequences from test data
        :param seq_len:
        :param batch_size:
        :return:
        """
        raise NotImplementedError()

    def get_train_batch_from_all_user(self, seq_len, batch_size):
        """
        Use last rating as test set, randomly select sequences from train set
        for example: if a user's rating sequence = [1, 3, 2, 3, 1, 5]
            train:  input = [1, 3, 2], label = [3]
                    input = [3, 2, 3], label = [1]

        :param seq_len:
        :param batch_size:
        :return:
        """
        raise NotImplementedError()

    def get_test_batch_from_all_user(self, seq_len, batch_size=None):
        """
        Use last rating as test set
        if batch_size is None, return all user's sequence
        :param seq_len:
        :param batch_size:
        :return:
        """
        raise NotImplementedError()
