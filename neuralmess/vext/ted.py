"""

 2019 (c) piteren

 Tokens Embeddings Dictionary (TED)

 TED object manages Dictionary of words
 - each word has assigned unique ID (int)
 - each word may have embedding (vector of floats) - this is optional

 There may be special words in Dictionary:
    >pad< - padding
    >unk< - unknown word
    >msk< - masked word
    >eot< - end of text (end of paragraph)

"""

import random

from ptools.neuralmess.vext.fastText.fasttextVectorManager import FTVec
from ptools.neuralmess.vext.gpt_encoder.bpencoder import BPEncoder

class TED:

    def __init__(
            self,
            iEmbO : FTVec or BPEncoder=      None,
            verb=                            0):

        self.verb = verb
        if self.verb > 0: print('\n*** TED *** Initializing...')

        if iEmbO is None:
            if self.verb > 0: print('Using default FTVec')
            iEmbO = FTVec(verb=verb)

        self.padID = 0
        self.unkID = 1
        self.mskID = 2
        self.eotID = 3

        self._ITS = {} # dictionary int >> string
        self._ITV = {} # dictionary int >> vec

        if type(iEmbO) is FTVec:
            self._ITS = iEmbO.getITS()
        else:
            pass

    # returns random word from dictionary
    def getRandomWord(self):
        return self._ITS[random.randrange(len(self._ITS))]


if __name__ == '__main__':

    ted = TED(verb=1)