"""

 https://www.tensorflow.org/hub/basics
 https://tfhub.dev/google/universal-sentence-encoder/3
 https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/semantic_similarity_with_tf_hub_universal_encoder.ipynb

"""

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece
from tqdm import tqdm

from putils.neuralmess.dev_manager import get_cuda_mem, tf_devices

# Universal Sentence Encoder
class UnSeEn:

    UM_001 = 'https://tfhub.dev/google/universal-sentence-encoder-large/3' # large version of first USE
    UM_002 = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1'  # large version of USE Multiling

    def __init__(
            self,
            pack_size :int=     0,      # USE embedding pack size (num of sentences in one interation), zero sets automatic
            device=             '/device:GPU:0',
            to_float16=         False,  # returns data reduced to np.float16
            verb=               0):

        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.verb = verb
        if self.verb > 0: print('\n*** UnSeEn *** loading...')

        self.to_float16 = to_float16
        self.pack_size = pack_size
        pack_auto = False
        if self.pack_size==0:
            pack_auto = True
            dev_mem_size = get_cuda_mem()
            self.pack_size = int(7000 * dev_mem_size/11171) # automatic estimation based on Ti1080 experience with USE
        if self.verb > 0: print(' > pack_size: %d (set auto: %s)'%(self.pack_size,pack_auto))

        device = tf_devices(device, verb=verb)[0]

        self.graph = tf.Graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            with tf.device(device):
                modelV = UnSeEn.UM_002
                if self.verb > 0: print(' > model: %s' % modelV)
                self.embedder = hub.Module(modelV)
                self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

        if self.verb > 0: print(' > initialized...')

    # returns list of embeddings(512) for list of sentences
    def make_emb(
            self,
            textList :list):

        if self.verb > 0: print(' > embedding %d sentences in packs of %d' % (len(textList), self.pack_size))

        subList = []
        listOfSenList = []
        for tx in textList:
            if len(subList) < self.pack_size:
                subList.append(tx)
            else:
                listOfSenList.append(subList)
                subList = [tx]
        listOfSenList.append(subList)

        emb_list = []
        with self.graph.as_default():
            iterable = tqdm(listOfSenList) if self.verb > 0 else listOfSenList
            for textL in iterable:
                emb_sl = np.split(self.session.run(self.embedder(textL)),len(textL))
                if self.to_float16: emb_sl = [np.squeeze(el).astype('float16') for el in emb_sl]
                else:               emb_sl = [np.squeeze(el) for el in emb_sl]
                emb_list += emb_sl
        return emb_list

    def close(self): self.session.close()


if __name__ == '__main__':

    # do some tests

    unseen = UnSeEn(device=0)

    sentences = [
        'The quick brown fox jumps over the lazy dog.',
        'I am a sentence for which I would like to get its embedding',
        '']

    emb = unseen.make_emb(sentences)
    for e in emb: print(e)

    unseen.close()
