"""

 2018 (c) piteren

"""

import heapq
import numpy as np
import os
import scipy.spatial

import ptools.lipytools.little_methods as lim
from ptools.textool.tokenization import tokenize_words


# fastText vector manager
class FTVec:

    DEF_VEC = 'crawl-300d-100K.vec'

    # reads vectors from file
    def __init__(
            self,
            fName=      None,   # vec filename
            useOOV=     True,   # add vector for OOV words (random)
            usePAD=     True,   # add vector for padding purposes (zeroes)
            verbLev=    0):

        self.verbLev = verbLev

        self._useOOV = useOOV
        self._usePAD = usePAD

        if self.verbLev > 0: print('\nFTVec inits (useOOV %s, usePAD %s)'%(self._useOOV,self._usePAD))

        if fName is None:
            if self.verbLev > 0: print('using default VEC file: %s' % FTVec.DEF_VEC)
            path = os.path.dirname(os.path.realpath(__file__))
            fName = path + '/' + FTVec.DEF_VEC
            assert os.path.isfile(fName[:-4] + '.dicts'), 'ERR: default .dicts for VEC file does not exists!'

        pickleDictFN = fName[:-4] + '.dicts'
        pickle = lim.r_pickle(pickleDictFN)
        if pickle:
            vec, vecSTI, vecITS = pickle
            if self.verbLev: print(' > got VEC from .dict file')
        # read VEC and save .dict
        else:
            if self.verbLev: print(' > builds VEC...')
            with open(fName, 'r') as file: fileLines = [line[:-1] for line in file]
            vec = []
            vecSTI = {}
            vecITS = {}
            ix = 0
            if len(fileLines[0].split())==2: fileLines=fileLines[1:] # trim first line (from FT .vec file)
            for line in fileLines:
                #split = line[:-1].split()
                split = line.split()
                if len(split)==301: # skip words with spaces
                    vec.append([float(x) for x in split[1:]])
                    vecITS[ix] = split[0]
                    vecSTI[split[0]] = ix
                    ix += 1

            if 'not' in vecSTI and 'n\'t' not in vecSTI: vecSTI['n\'t'] = vecSTI['not'] # add n't to dictionary

            vecWidth = len(vec[0])
            oovVEC = np.random.normal(loc=0.0, scale=0.1, size=[1, vecWidth]).astype(dtype=np.float16)
            padVEC = np.zeros(shape=[1, vecWidth], dtype=np.float16)
            vec = np.asarray(vec, dtype=np.float16)
            vec = np.concatenate([vec,oovVEC,padVEC])

            lim.w_pickle((vec, vecSTI, vecITS), pickleDictFN)

        self._vecARR = vec
        self._vecSTI = vecSTI # string to int
        self._vecITS = vecITS # int to string

        self.vecNum = self._vecARR.shape[0] # num of vectors
        self.vecLen = self._vecARR.shape[1] # vector length

        self.oovID = self.vecNum-2 if self._useOOV else None
        self.padID = self.vecNum-1 if self._usePAD else None
        if self.verbLev: print(' > FTVec got %d vec of width %d' % (self.vecNum, self.vecLen))

    def getITS(self): return self._vecITS

    # returns int (ID) for given word
    def int(self, word):
        int = None
        if word in self._vecSTI: int = self._vecSTI[word]
        if int is None and self._useOOV: int = self.oovID
        return int

    # returns vector for given word
    def vec(self, word):
        vecID = self.int(word)
        vec = self._vecARR[vecID] if vecID is not None else None
        return vec

    # returns distance 0-2 between two words or vectors
    def dist(self, wa, wb):
        if type(wa) is str: wa = self.vec(wa)
        if type(wb) is str: wb = self.vec(wb)
        return scipy.spatial.distance.cosine(wa, wb)

    # returns similarity 1-0 of two words or vectors
    def simm(self, wa, wb):
        if type(wa) is str: wa = self.vec(wa)
        if type(wb) is str: wb = self.vec(wb)
        return 1 - self.dist(wa, wb) / 2

    # returns list of n closest (dist,ix,vec) to given word/vector
    # TODO: check and refactor this method!!!
    def closest(
            self,
            wa,
            n=      10):

        if type(wa) is str: wa = self.vec(wa)

        h = [] # heap
        maxFromN = 0 # max distance of n words/vectors
        for ix in range(self.vecNum):
            vecA = self._vecARR[ix]
            dist = self.dist(wa, vecA)
            if len(h) < n:
                heapq.heappush(h, (dist, ix, vecA))
                if dist > maxFromN: maxFromN = dist
            else:
                if dist < maxFromN:
                    heapq.heappush(h, (dist, ix, vecA))
                    nh = []
                    for _ in range(n):
                        el = heapq.heappop(h)
                        maxFromN = el[0]
                        heapq.heappush(nh, el)
                    h = nh
        closest = [heapq.heappop(h) for _ in range(n)]
        return closest

    # returns list of token sequences for given list of sentences
    def tksFromSL(
            self,
            senL :list,
            tokenizer=      None, # tokenization method
            limitNT=        None, # int, upper limit of num tokens in seq (before padding), trim if exceeded
            padTo=          None): # False - do not pads, None - pads to amax(nwTokens) or limitNT, int - pads to int,

        if not tokenizer: tokenizer = tokenize_words

        # stats
        nTokens = []  # seq of num of tokens in sentence
        oov = {}  # dict {'oovToken': n} n - occurrence

        tksL = []
        for sen in senL:
            wordTokens = tokenizer(sen)
            tks = []
            for tok in wordTokens:
                tint = self.int(tok)
                if tint==self.oovID or tint is None: oov[tok] = oov[tok] + 1 if tok in oov else 1
                if tint is not None: tks.append(tint)
            nTokens.append(len(tks))
            if limitNT: tks = tks[:limitNT]
            tksL.append(tks)

        # print stats
        if self.verbLev:
            nTokens = np.asarray(nTokens)
            print('### num of tokens/sen')
            print(' min :      %d' % np.amin(nTokens))
            print(' max :      %d' % np.amax(nTokens))
            print(' mean:      %.1f' % np.mean(nTokens))
            print(' std :      %.1f' % np.std(nTokens))
            print(' oov/total: %.1f%%' % (100 * sum(oov.values()) / sum(nTokens)))

        if self.verbLev > 1:
            oovInv = {key: [] for key in set(oov.values())}
            for key in oov.keys(): oovInv[oov[key]].append(key)
            for key in oovInv.keys(): oovInv[key] = sorted(oovInv[key])
            for key in sorted(list(oovInv.keys()), reverse=True):
                print(key, end=' ')
                for w in oovInv[key]:
                    print(w, end=' ')
                print()

        # padding
        if padTo is not False:
            if padTo is None:
                padTo = np.amax(nTokens)
                if limitNT and padTo > limitNT: padTo = limitNT
            for tks in tksL:
                while len(tks) < padTo: tks.append(self.padID)

        return tksL

    # returns list of vector sequences for given list of sentences
    def seqFromSL(self):
        pass

# prepares sequence of fastText vectors (embeddings) sequences for given seq of sentences
def ftvsFromSenSeq(
        senSeq,
        ftVEC :FTVec,
        tokenizer=      None,
        oovToRnd=       True,   # put OOV as rndVEC
        padTo=          None,   # False - do not pads, None - pads to amax(nwTokens), int - pads to int,
        verbLev=        0):

    if not tokenizer: tokenizer = tokenize_words

    nwTokens = []       # seq of num of word tokens in sentence
    oov = {}            # dict {'oovToken': n} n - occurrence
    seqSenWEmb = []     # seq of seq of tokenEmb (with None)
    for sen in senSeq:
        wordTokens = tokenizer(sen)
        nwTokens.append(len(wordTokens))
        senWEmb = []
        for tok in wordTokens:
            tEmb = ftVEC.vec(tok)
            if tEmb is None:
                oov[tok] = oov[tok] + 1 if tok in oov else 1
                if oovToRnd: tEmb = ftVEC.rndLST
            if tEmb is not None: senWEmb.append(tEmb)
        seqSenWEmb.append(senWEmb)

    # print stats
    if verbLev > 0:
        nwTokens = np.asarray(nwTokens)
        print('### num of tokens/sen')
        print(' min :      %d' % np.amin(nwTokens))
        print(' max :      %d' % np.amax(nwTokens))
        print(' mean:      %.1f' % np.mean(nwTokens))
        print(' std :      %.1f' % np.std(nwTokens))
        print(' oov/total: %.1f%%' % (100 * sum(oov.values()) / sum(nwTokens)))

    if verbLev > 1:
        oovInv = {key: [] for key in set(oov.values())}
        for key in oov.keys(): oovInv[oov[key]].append(key)
        for key in oovInv.keys(): oovInv[key] = sorted(oovInv[key])
        for key in sorted(list(oovInv.keys()), reverse=True):
            print(key, end=' ')
            for w in oovInv[key]:
                print(w, end=' ')
            print()

    # padding
    if padTo is not False:
        if padTo is None: padTo = np.amax(nwTokens)
        for senWEmb in seqSenWEmb:
            while len(senWEmb) < padTo:
                senWEmb.append(ftVEC.zerosVec)

    return seqSenWEmb


if __name__ == '__main__':

    """
    #fName = '../_Gdata/fastText/crawl-300d-100K.vec'
    fName = '../_Gdata/glove/glove.840B.300d.txt'
    with open(fName, 'r') as file: fileLines = [line[:-1] for line in file]
    print(len(fileLines))
    """

    ftv = FTVec(verbLev=2)