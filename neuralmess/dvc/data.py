"""

 2019 (c) piteren

"""

import numpy as np
import os
import random

from putils.lipytools.little_methods import r_pickle, w_pickle
from putils.neuralmess.vext.USE import UnSeEn
from putils.neuralmess.vext.fastText.fasttextVectorManager import FTVec
from putils.neuralmess.vext.gpt_encoder.bpencoder import BPEncoder, get_encoder


# calculates sizes of DATA_PARTS in data
def data_size(data):
    ds = {PT: 0 for PT in DVCData.DATA_PARTS}
    for PT in ds:
        for tp in DVCData.DATA_TYPES:
            if tp != 'lbl' and len(data[PT+tp]):    # not empty tuple of not lbl
                ds[PT] = len(data[PT+tp][0])        # length of list in tuple
                break
    return ds

# sets new split (TR >> VL & TS) for given data (...makes copy into new dict)
def split(
        data,
        seed :int,
        vl_split=   0.0,
        ts_split=   0.1,
        verb=       0):

    random.seed(seed)

    ds = data_size(data)
    n_classif = len(data['TRlbl'])

    # calc num of samples per part
    nVL = int(ds['TR'] * vl_split)
    nTS = int(ds['TR'] * ts_split)
    nTR = ds['TR'] - nVL - nTS
    if verb > 0: print('Setting new split of data: TR %d, VL %d, TS %d (seed %d)'%(nTR, nVL, nTS, seed))

    # prepare data_split_code (list of data PTs)
    data_split_code = ['TR'] * nTR + ['TS'] * nTS + ['VL'] * nVL
    random.shuffle(data_split_code)

    # struct below will store new distribution of training data, by now lists
    new_dist = {}
    for PT in DVCData.DATA_PARTS:
        new_dist.update({PT+tp: [] for tp in DVCData.DATA_TYPES})

    # split user 'TR' among newDist
    for ix in range(ds['TR']): # iterate over TR
        target_PT = data_split_code[ix] # target part
        for tp in DVCData.DATA_TYPES: # iterate over data types
            if tp!='lbl':
                if not len(new_dist[target_PT+tp]): new_dist[target_PT+tp] = [[] for _ in data['TR'+tp]] # create nested list (multi_sen)
                for lix in range(len(data['TR'+tp])): # iterate over lists of tuple
                    new_dist[target_PT+tp][lix].append(data['TR'+tp][lix][ix])
            # labels
            else:
                if not len(new_dist[target_PT+'lbl']):
                    new_dist[target_PT+'lbl'] = [[]]*n_classif  # create nested list
                    #print(target_PT, new_dist[target_PT+'lbl'])
                for cix in range(n_classif):
                    new_dist[target_PT+'lbl'][cix].append(data['TRlbl'][cix][ix])

    # if there was VL, TS or IF given >> copy it
    for PT in ['VL','TS','IF']:
        if ds[PT]:
            for tp in DVCData.DATA_TYPES:
                if tp != 'lbl':
                    if len(data[PT+tp]):
                        new_dist[PT+tp] = [ls for ls in data[PT+tp]]
                else: new_dist[PT+'lbl'] = data[PT+'lbl']

    # change list to tuples
    for PT in DVCData.DATA_PARTS:
        for tp in DVCData.DATA_TYPES:
            if tp != 'lbl':
                new_dist[PT+tp] = tuple(new_dist[PT+tp])

    return new_dist


# user data dict, is early structure(dict) of data to be filled by user >> goes to DVCData
class UDD(dict):

    # [sen vec tks seq] data types may be given as list or tuple of lists (for multi_sen)
    # labels may be given as list(of labels) or list of lists(sample_labels = multi-classif per sample)
    # VL part may be split from TR (here or in DVCdata)
    def __init__(
            self,
            # train
            TRsen: list or tuple =  None,
            TRvec: list or tuple =  None,
            TRtks: list or tuple =  None,
            TRseq: list or tuple =  None,
            TRlbl: list =           None,
            # test
            TSsen: list or tuple =  None,
            TSvec: list or tuple =  None,
            TStks: list or tuple =  None,
            TSseq: list or tuple =  None,
            TSlbl: list =           None,
            # inference
            IFsen: list or tuple =  None,
            IFvec: list or tuple =  None,
            IFtks: list or tuple =  None,
            IFseq: list or tuple =  None):

        super().__init__()

        self['TRsen'] = TRsen
        self['TRvec'] = TRvec
        self['TRtks'] = TRtks
        self['TRseq'] = TRseq
        self['TRlbl'] = TRlbl

        self['VLsen'] = None
        self['VLvec'] = None
        self['VLtks'] = None
        self['VLseq'] = None
        self['VLlbl'] = None

        self['TSsen'] = TSsen
        self['TSvec'] = TSvec
        self['TStks'] = TStks
        self['TSseq'] = TSseq
        self['TSlbl'] = TSlbl

        self['IFsen'] = IFsen
        self['IFvec'] = IFvec
        self['IFtks'] = IFtks
        self['IFseq'] = IFseq
        self['IFlbl'] = None # add label for smarter processing

        self.__data_check()

        # convert data types into proper format:
        # - tuples of lists for all keys but lbl
        # - lists of lists or empty list for lbl
        for key in self.keys():
            if 'lbl' not in key:
                if type(self[key]) is list: self[key] = (self[key],)            # enclose list into tuple
                if self[key] is None:       self[key] = ()                      # empty tuple
                # assumes proper format if given as tuple
            elif not self[key]:             self[key] = []                      # empty labels list (for no lables)

        self.__invert_labels()

    # data consistency check
    def __data_check(self):
        # TODO
        # check length
        # check same data types across PTs
        # check tuple length
        # check labels multi num
        pass

    # prepares inverted labels for TR VL TS
    # by now labels is list of lables or LL: [[labels of sample]*n_samples] [[0,0],[0,1],[1,0]...]
    # below we are going to invert indexing: [[labels of classif] x n_classif] [[0,0,1..][0,1,0..]]
    def __invert_labels(self):
        inv_parts = ['TR','VL','TS']
        for PT in inv_parts:
            if self[PT+'lbl']: # got PT labels
                if type(self[PT+'lbl'][0]) is list:
                    inv_labels = [[] for _ in self[PT+'lbl'][0]]
                    for lab in self[PT+'lbl']:
                        for ix in range(len(lab)):
                            inv_labels[ix].append(lab[ix])
                    self[PT+'lbl'] = inv_labels # put inverted
                else: self[PT+'lbl'] = [self[PT+'lbl']] # enclose only

    # splits TS & VL from TR as it would be given by user (fixes TS & VL for DVCData)
    def early_split(self, seed, vl_split=0.0, ts_split=0.1):
        self.update(split(self,seed,vl_split,ts_split))

# keeps data for DVC
class DVCData:

    # validation PT may be split ONLY from training data
    # test PT may be split from training data or given explicitly
    DATA_PARTS = ['TR','VL','TS','IF'] # training, validation, test, inference
    DATA_TYPES = ['sen','vec','tks','seq','lbl'] # sentences, vectors, tokensS, vectorS, labels

    # TODO: replace use, FT.. with TED
    def __init__(
            self,
            uDD: UDD=                       None,   # for None uses ONLY cache
            # preparation settings
            merge_multisen=                 False,  # merges uDD multi-sen into one sequence
            seed=                           12321,  # seed for random shuffle of data (for data distribution)
            vl_split=                       0.1,    # VL data split (from TR)
            ts_split=                       0.1,    # TS data split (from TR)
            cache_file=                     None,   # path to cache file
            use: UnSeEn or bool =           None,   # object or True (for default)
            ftVEC: FTVec or bool=           None,   # object or True (for default)
            bpeENC: BPEncoder or bool =     None,   # object or True (for default)
            limitNT=                        None,   # upper limit of num tokens in seq (FT,BPE) before padding, if exceeded then trim
            # TODO: mergeAB=                None,   # token int, merges tokens sequences into one (with token separator)
            verb=                           0):

        self.verb = verb
        if self.verb > 0: print('\n*** DVCdata *** initializes...')

        assert not (ftVEC and bpeENC), 'Error, cannot process texts with FT and BPE!'

        if cache_file and os.path.isfile(cache_file):
            if self.verb > 0: print(' > loading the data from cache (%s)...' % cache_file)
            uDD, lbl_dictL = r_pickle(cache_file) # load already preprocessed uDD
        # preprocess uDD and save cache
        else:
            if self.verb > 0: print(' > preprocessing of uDD...')

            # gather all texts
            all_texts = []
            for key in uDD.keys():
                if 'sen' in key and len(uDD[key]): # every key of sentences
                    for ls in uDD[key]: # every list from tuple
                        all_texts += ls

            # process texts with USE,FT,BPE
            if all_texts:
                if self.verb > 0: print(' > got %d sentences from uDD' % len(all_texts))
                # default encoders
                if use is True: use = UnSeEn(verb=verb)
                if ftVEC is True: ftVEC = FTVec()
                if bpeENC is True: bpeENC = get_encoder()

                text_vecs = [] # here put (single)vectors from sentences
                if use:
                    if self.verb > 0: print(' > encoding with USE >> vec')
                    text_vecs = use.make_emb(all_texts)

                text_toks = [] # here put token sequences from sentences
                if ftVEC:
                    if self.verb > 0: print(' > encoding with FT >> tks')
                    text_toks = ftVEC.tksFromSL(senL=all_texts, limitNT=limitNT, padTo=None)
                if bpeENC:
                    if self.verb > 0: print(' > encoding with BPE >> tks')
                    padID = len(bpeENC.decoder) - 1
                    maxLen = 0
                    for txt in all_texts:
                        tks = bpeENC.encode(txt)
                        if limitNT: tks = tks[:limitNT]
                        if maxLen < len(tks): maxLen = len(tks)
                        text_toks.append(tks)
                    for tks in text_toks:
                        while len(tks) < maxLen: tks.append(padID)

                # split generated vec or tks among uDD keys
                for key in uDD.keys():
                    if 'sen' in key:
                        for ls in uDD[key]:
                            cut = len(ls)
                            if text_vecs:
                                nowKey = key[:-3] + 'vec'
                                if type(uDD[nowKey]) is tuple: uDD[nowKey] = [] # convert to list (temporary)
                                uDD[nowKey].append(text_vecs[:cut])
                                text_vecs = text_vecs[cut:]
                            if text_toks:
                                nowKey = key[:-3] + 'tks'
                                if type(uDD[nowKey]) is tuple: uDD[nowKey] = [] # convert to list (temporary)
                                uDD[nowKey].append(text_toks[:cut])
                                text_toks = text_toks[cut:]

                # convert lists back to tuples
                for key in uDD.keys():
                    if 'lbl' not in key:
                        if type(uDD[key]) is list: uDD[key] = tuple(uDD[key])

            # prepare lbl_dictL [list (per classifier) of labels dictionaries (list of dictionaries) that translate each label to int <0,n-1>]
            lbl_dictL = [{lab: x for x, lab in enumerate(sorted(set(c_lab)))} for c_lab in uDD['TRlbl']] if uDD['TRlbl'] else None
            # and translate
            for key in uDD.keys():
                if 'lbl' in key:  # label key
                    if uDD[key]:
                        trans_lables = []
                        for cix in range(len(lbl_dictL)):
                            trans_lables.append([lbl_dictL[cix][lbl] for lbl in uDD[key][cix]])
                        uDD[key] = trans_lables

            if cache_file: w_pickle((uDD,lbl_dictL), cache_file) # save preprocessed uDD

        self.uDD = uDD
        self.lbl_dictL = lbl_dictL

        self.uDD_size = data_size(uDD) # data length
        if self.verb > 0:
            for PT in DVCData.DATA_PARTS: print(' >> got %s data of %d samples' % (PT, self.uDD_size[PT]))

        self.vl_split = vl_split
        self.ts_split = ts_split
        if self.uDD_size['VL']: self.vl_split = 0  # in case of explicit given VL do not split it from train
        if self.uDD_size['TS']: self.ts_split = 0  # in case of explicit given TS do not split it from train

        # resolve present data types
        self.got_sen = False # got sentences
        self.got_vec = False # got vector
        self.got_tks = False # got tokens sequence
        self.got_seq = False # got vectors sequence
        for PT in DVCData.DATA_PARTS:
            for tp in DVCData.DATA_TYPES:
                if tp!='lbl':
                    if PT+tp in uDD:
                        if tp == 'sen' and uDD[PT+tp]: self.got_sen = True
                        if tp == 'vec' and uDD[PT+tp]: self.got_vec = True
                        if tp == 'tks' and uDD[PT+tp]: self.got_tks = True
                        if tp == 'seq' and uDD[PT+tp]: self.got_seq = True
        if self.verb > 0:
            print(' > data types present:')
            if self.got_sen: print(' >> sen (sentences)')
            if self.got_vec: print(' >> vec (vector)')
            if self.got_tks: print(' >> tks (sequence of tokens)')
            if self.got_seq: print(' >> seq (sequence of vectors)')

        # resolve multi_sen
        self.multi_sen = 0
        for key in uDD.keys():
            if type(uDD[key]) is tuple:
                if len(uDD[key]) > 0:
                    self.multi_sen = len(uDD[key])
                    break
        if self.verb > 0: print(' > data multi-sen (tuple len): %d' % self.multi_sen)

        # merge multi-sen
        if self.multi_sen>1 and merge_multisen:
            if self.verb > 0: print(' > merging multi-sen...')
            for key in uDD.keys():
                if key != 'sen' and type(uDD[key]) is tuple:
                    uDD[key] = ([np.concatenate([uDD[key][eix][six] for eix in range(len(uDD[key]))], axis=0) for six in range(len(uDD[key][0]))],) # tuple with list of concatenated multi-sen samples over time(0) axis
            self.multi_sen = 1

        # report labels distribution
        if self.verb > 0 and self.lbl_dictL:
            inv_parts = ['TR','TS']
            for PT in inv_parts:
                if uDD[PT+'lbl']:
                    print(' > got %s labels of %d classifiers with distribution:' % (PT,len(self.lbl_dictL)))
                    for cix in range(len(self.lbl_dictL)):
                        print(' >> classifier %d' % cix)
                        clD = self.lbl_dictL[cix]
                        inv_clD = {clD[key]: key for key in clD} # inverted dictionary of classifier labels
                        labDist = {lab: 0 for lab in sorted(list(clD.keys()))}
                        for lab in uDD[PT+'lbl'][cix]: labDist[inv_clD[lab]] += 1
                        sum = len(uDD[PT+'lbl'][cix])
                        for lab in sorted(list(clD.keys())):
                            labDist[lab] = labDist[lab] / sum * 100
                            print(' >>> label: %d - (%.1f%%) [original label: %s]' %(clD[lab], labDist[lab], lab))

        self.tdata = None
        self.data_dist_seed = None
        self.new_data_distribution(seed)

    # randomly (with given seed) distribute TR data among TR VL TS
    def new_data_distribution(
            self,
            seed: int,              # seed to split the data with
            force_silent=   False): # forces method to be silent

        new_dist = {}
        if self.data_dist_seed!=seed and self.uDD_size['TR'] and self.vl_split+self.ts_split>0: # only if needed
            new_dist = split(
                data=       self.uDD,
                seed=       seed,
                vl_split=   self.vl_split,
                ts_split=   self.ts_split,
                verb=       self.verb if not force_silent else 0)
        else:
            if self.verb>0 and not force_silent: print(' > not splitting (new split conditions not meet...)')

        self.data_dist_seed = seed
        if new_dist:            self.tdata = new_dist
        elif not self.tdata:    self.tdata = self.uDD # copy uDD to self.tdata (case of new distribution without splitting ONLY)