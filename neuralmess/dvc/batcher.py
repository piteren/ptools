"""

 2018 (c) piteren

 DVCbatcher
    - prepares batches of data (from DVCdata)

"""

import random

from ptools.neuralmess.dvc.data import DVCData


# DVC Batcher (for DVCdata)
class DVCBatcher:

    def __init__(
            self,
            dvc_data :DVCData,
            seed=           12321,
            batch_size=     500,
            bsm_VL=         2,      # multiplier of VL and TS batchSize (since no backprop.)
            bsm_IF=         4,      # multiplier of IF batchSize
            random_TR=      None,   # None sets automatic random (based on size of data)
            verb=           0):

        self.verb = verb
        self.seed_cnt = seed # seed counter

        self.dvc_data = dvc_data
        self.data_IX = {PT: 0 for PT in DVCData.DATA_PARTS} # index (pointer) for every PART

        # calculate sizes
        self.data_size = {PT: 0 for PT in DVCData.DATA_PARTS}
        for PT in DVCData.DATA_PARTS:
            for tp in DVCData.DATA_TYPES:
                if tp!='lbl' and self.dvc_data.tdata[PT+tp]:
                    self.data_size[PT] = len(self.dvc_data.tdata[PT+tp][0])
                    break

        self.target_batch_size = {
            'TR': batch_size,
            'VL': batch_size*bsm_VL,
            'TS': batch_size*bsm_VL,
            'IF': batch_size*bsm_IF}

        # resolve automatic (random) batch for TR
        self.random_TR = random_TR
        if self.random_TR is None:
            if self.data_size['TR'] < 1e6: self.random_TR = False
            else: self.random_TR = True

        if self.verb > 0:
            print('\n*** DVCBatcher *** initialized (seed %d, bS %d)...' % (self.seed_cnt, batch_size))
            print(' > random data sampling (training):', self.random_TR)
            for part in DVCData.DATA_PARTS: print(' >> %s size: %d' % (part, self.data_size[part]))

    # returns next batch, for VL, TS, IF returns None every loop
    def get_batch(self, part='TR'):

        # next loop (possible only for not TR)
        if self.data_IX[part]==self.data_size[part]:
            self.data_IX[part] = 0
            return None

        batch = {tp: [] for tp in DVCData.DATA_TYPES}
        batch_size = 0

        used_IX = [] # set of sample indexes @batch (to prevent duplicates)
        random.seed(self.seed_cnt)
        self.seed_cnt += 1
        while True:

            # resolve index sample (avoiding duplicates)
            if part == 'TR' and self.random_TR:
                while True:
                    ix = random.randrange(self.data_size[part])
                    if ix not in used_IX:
                        used_IX.append(ix)
                        break
            # not TR
            else:
                ix = self.data_IX[part] 
                self.data_IX[part] += 1

            # copy data
            for tp in batch.keys():
                if self.dvc_data.tdata[part + tp]:
                    for lix in range(len(self.dvc_data.tdata[part + tp])):
                        if not batch[tp]: batch[tp] = [[] for _ in self.dvc_data.tdata[part + tp]]
                        batch[tp][lix].append(self.dvc_data.tdata[part + tp][lix][ix])
            batch_size += 1

            if self.data_IX[part]==self.data_size[part]:
                if part == 'TR': self.data_IX[part] = 0 # loop for: TR_but_not_random
                else: break

            if batch_size == self.target_batch_size[part]: break

        return batch
