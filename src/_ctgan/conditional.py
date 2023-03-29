import numpy as np


class ConditionalGenerator(object):
    """A class that generates conditional data based on the given input data and output information.

    Args:
        data (numpy.ndarray): The input data.
        output_info (list): A list of tuples containing information about the output data.
        log_frequency (bool): A boolean value indicating whether to use logarithmic frequency.

    Attributes:
        model (list): A list of models.
        interval (numpy.ndarray): An array of intervals.
        n_col (int): The number of columns.
        n_opt (int): The number of options.
        p (numpy.ndarray): An array of probabilities.
    """
    def __init__(self, data, output_info, log_frequency):
        self.model = []

        start = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                skip = True
                continue

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    start += item[0]
                    continue

                end = start + item[0]
                max_interval = max(max_interval, end - start)
                counter += 1
                self.model.append(np.argmax(data[:, start:end], axis=-1))
                start = end

            else:
                raise AssertionError

        if start != data.shape[1]:
            raise AssertionError

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        start = 0
        self.p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                start += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    start += item[0]
                    skip = False
                    continue
                end = start + item[0]
                tmp = np.sum(data[:, start:end], axis=0)
                if log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                start = end
            else:
                raise AssertionError

        self.interval = np.asarray(self.interval)

    def random_choice_prob_index(self, idx):
        """Randomly selects an index based on the given probabilities.
        Args:
            idx (numpy.ndarray): An array of indices.
        Returns:
            numpy.ndarray: An array of randomly selected indices.
        """
        a = self.p[idx]
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample(self, batch):
        """Samples data based on the given batch size.
        Args:
            batch (int): The batch size.
        Returns:
            tuple: A tuple containing the generated data, mask, index, and option.
        """
        if self.n_col == 0:
            return None

        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = self.random_choice_prob_index(idx)
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

    def sample_zero(self, batch):
        """Samples zero data based on the given batch size.
        Args:
            batch (int): The batch size.
        Returns:
            numpy.ndarray: An array of generated zero data.
        """
        if self.n_col == 0:
            return None

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1

        return vec
