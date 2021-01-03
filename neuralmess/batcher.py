"""

 2020 (c) piteren

    NN Batcher

"""

import numpy as np

BTYPES = [
    'base',         # prepares batches in order of given data
    'random',       # basic random sampling
    'random_cov']   # random sampling with full coverage of data


class Batcher:


    def __init__(
            self,
            data: dict,                 # data in dict of lists or np.arrays
            batch_size: int= None,
            btype: str=     'random_cov',
            seed=           123,
            verb=           0):

        self.verb = verb
        self.seed_counter =seed

        assert btype in BTYPES, f'ERR: unknown btype, possible: {BTYPES}'
        self.btype = btype

        self._batch_size = batch_size
        self._data_keys = sorted(list(data.keys()))
        self._data = {k: np.asarray(data[k]) for k in self._data_keys} # as numpy
        self._data_len = self._data[self._data_keys[0]].shape[0]

        self._data_ixmap = []

        if verb>0:
            print(f'\nBatcher initialized with {self._data_len} samples of data in keys:')
            for k in self._data_keys: print(f' > {k} ({type(data[k][0])})')
            print(f' batch size: {batch_size}')


    def _extend_ixmap(self):

        if self.btype == 'base':
            self._data_ixmap += list(range(self._data_len))

        if self.btype == 'random':
            self._data_ixmap += np.random.choice(self._data_len, self._batch_size, replace=False).tolist()

        # TODO: current procedure does not guarantee full coverage in every loop (edges case - to solve)
        if self.btype == 'random_cov':
            new_ixmap = np.arange(self._data_len)
            np.random.shuffle(new_ixmap)
            new_ixmap = new_ixmap.tolist()
            while len(self._data_ixmap) < self._batch_size:
                n_misses = self._batch_size - len(self._data_ixmap)
                candidates = new_ixmap[:n_misses]
                new_ixmap = new_ixmap[n_misses:]
                # add only those not present in current self._data_ixmap (to avoid replacement in a batch)
                for c in candidates:
                    if c not in self._data_ixmap: self._data_ixmap.append(c)
                    else: new_ixmap.append(c)
            self._data_ixmap += new_ixmap

    def set_batch_size(self, bs: int): self._batch_size = bs

    def get_batch(self):

        # set seed
        np.random.seed(self.seed_counter)
        self.seed_counter += 1

        if len(self._data_ixmap) < self._batch_size: self._extend_ixmap()
        
        indexes = self._data_ixmap[:self._batch_size]
        self._data_ixmap = self._data_ixmap[self._batch_size:]
        return {k: self._data[k][indexes] for k in self._data_keys}

# test for coverage of batchers
def test_coverage(btype):

    print(f'\nStarts coverage of {btype}')

    c_size = 1000
    b_size = 64

    samples = np.arange(c_size)
    np.random.shuffle(samples)
    samples = samples.tolist()

    labels = np.random.choice(2, c_size).tolist()

    data = {
        'samples': samples,
        'labels': labels}

    batcher = Batcher(data, b_size, btype=btype, verb=1)

    for _ in range(10):
        sL = []
        n_b = 0
        while True:
            bs = batcher.get_batch()['samples']
            n_b += 1
            sL += bs.tolist()
            if len(set(sL)) == c_size:
                print(f'got full coverage with {n_b} batches')
                break
    print(f' *** finished coverage tests')

# test for reproducibility of batcher
def test_seed():

    print(f'\nStarts seed tests')

    c_size = 1000
    b_size = 64

    samples = np.arange(c_size)
    np.random.shuffle(samples)
    samples = samples.tolist()

    labels = np.random.choice(2, c_size).tolist()

    data = {
        'samples': samples,
        'labels': labels}

    batcher = Batcher(data, b_size, btype='random_cov', verb=1)
    sA = []
    while len(sA) < 10000:
        sA += batcher.get_batch()['samples'].tolist()
        np.random.seed(len(sA))

    batcher = Batcher(data, b_size, btype='random_cov', verb=1)
    sB = []
    while len(sB) < 10000:
        sB += batcher.get_batch()['samples'].tolist()
        np.random.seed(10000000-len(sB))

    for ix in range(len(sA)):
        if sA[ix] != sB[ix]: print('wrong')
    print(f' *** finished seed tests')


if __name__ == '__main__':

    for cov in BTYPES: test_coverage(cov)
    test_seed()