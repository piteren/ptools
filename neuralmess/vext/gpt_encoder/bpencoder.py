"""

 2019 (c) piteren

 bpe encoder from GPT code

"""

import json
import regex as re
from functools import lru_cache
import numpy as np
import os
from tqdm import tqdm

from ptools.lipytools.little_methods import w_pickle, r_pickle


@lru_cache() # decorator that saves recent calls
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    # ? base strings - 188, ords of base char dict?
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    #print(bs)
    #print([chr(s) for s in bs])
    cs = bs[:]
    n = 0
    # appends to bs missing numbers from 0-255
    # appends for them strings from >255
    for b in range(2**8):   # 256
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]   # change ord to char
    return dict(zip(bs, cs))    # makes dict{bs: cs}


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPEncoder:

    def __init__(
            self,
            encoder,
            bpe_merges,
            distribution=   None,       # distribution of tokens [probs], for random token generation
            tok_len=        None,       # length of token
            errors=         'replace'):

        self.encoder = encoder                                          # dict of 50257 {str: int}
        self.decoder = {v:k for k,v in self.encoder.items()}            # dict of 50257 {int: str}
        self.distribution = distribution
        self.tok_len = tok_len
        self.errors = errors # how to handle errors in decoding

        self.byte_encoder = bytes_to_unicode()                          # dict{int: char}
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()} # dict{char: int}

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.pad_tok = len(self.decoder) - 1
        self.pad_str = self.decode([self.pad_tok])


    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:

            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks: break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    # encodes text (str) into bpe tokens
    def encode(self, text, progress=False):
        bpe_tokens = []
        if progress: print(' > regex search...')
        iterable = re.findall(self.pat, text)
        if progress:
            iterable = tqdm(iterable)
            print(' > regex ready')
        for token in iterable:
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    # decodes tokens into text (str)
    def decode(self, tokens):
        #if type(tokens) is not type(list): tokens = [tokens]
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    # returns random code
    def get_random_code(self):
        return np.random.choice(a=len(self.encoder), p=self.distribution)

# prepares distribution data (for constructor)
def tok_dist(enc: BPEncoder, filePath):

    with open(filePath, 'r') as file:
        lines = [line[:-1] for line in file]

    print(' > got %d lines of text, processing...' % len(lines))

    encLines = []
    for text in tqdm(lines): encLines.append(enc.encode(text))

    occDict = {x: 0 for x in range(len(enc.encoder))}
    for eLine in tqdm(encLines):
        for i in eLine:
            if i not in occDict:    occDict[i] = 1
            else:                   occDict[i] += 1

    occList = [occDict[k] for k in sorted(list(occDict.keys()))]
    sumOcc = sum(occList)
    dist = [el/sumOcc for el in occList]

    w_pickle(dist, os.path.dirname(os.path.realpath(__file__)) + '/enc.dist')

    return dist

# prepares token length dictionary (for constructor)
def tok_length(enc: BPEncoder):

    tLen = {ix : len(enc.decode([ix])) for ix in range(len(enc.decoder))}
    w_pickle(tLen, os.path.dirname(os.path.realpath(__file__)) + '/enc.tLen')
    return tLen

# return sencoder
def get_encoder(verbLev=0):

    path = os.path.dirname(os.path.realpath(__file__))

    with open(path + '/encoder.json', 'r') as f: encoder = json.load(f)
    with open(path + '/vocab.bpe', 'r', encoding="utf-8") as f: bpe_data = f.read()
    distribution = r_pickle(path + '/enc.dist')
    tok_len = r_pickle(path + '/enc.tLen')
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]

    enc = BPEncoder(
        encoder=        encoder,
        bpe_merges=     bpe_merges,
        distribution=   distribution,
        tok_len=        tok_len)

    if verbLev > 0:
        print('\nGot BPE Encoder (%s)' % path)
        print(' > encoder length', len(enc.decoder))
        print(' > distribution', type(distribution))
        print(' > tLen', type(tok_len))

    return enc


if __name__ == '__main__':

    enc = get_encoder()
    #makeBPEdist(enc, '../_Gdata/cnnLM/wortschatzSMALL/wortschatzCorp.txt')
    tok_length(enc)
    print(enc.pad_str)