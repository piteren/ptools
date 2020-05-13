"""

 2018 (c) piteren

"""

import nltk
import rouge

from ptools.textool.tokenization import tokenize_words
from ptools.neuralmess.vext.gpt_encoder.bpencoder import get_encoder


# returns Levenshtein-distance of two strings
def lev_dist(source :str, target :str):
    return nltk.edit_distance(source, target)

# returns Levenshtein-distance of two lists (or strings)
def lev_distL(source :list or str, target :list or str):

    # same
    if len(source) == len(target):
        if not sum([1 for e in zip(source,target) if e[0] == e[1]]):
            return 0

    # prepare a matrix
    slen, tlen = len(source), len(target)
    dist = [[0 for _ in range(tlen+1)] for _ in range(slen+1)]
    for i in range(slen+1): dist[i][0] = i
    for j in range(tlen+1): dist[0][j] = j

    # count distance
    for i in range(slen):
        for j in range(tlen):
            cost = 0 if source[i] == target[j] else 1
            dist[i + 1][j + 1] = min(dist[i][j + 1] + 1,    # delete
                                     dist[i + 1][j] + 1,    # insert
                                     dist[i][j] + cost      # substitute
            )
    return dist[-1][-1]

# returns bpe-tokens-Levenshtein-distance of two strings
def lev_dist_bpe(source :str, target :str, bpe_enc=None):
    if not bpe_enc: bpe_enc = get_encoder() # default (GPT) encoder
    return lev_distL(bpe_enc.encode(source),bpe_enc.encode(target))

# returns BLEU value
def bleu(
        sen :str,
        ref :list or str,
        ngram=      None,                   # set to 1,2,3,4 to override weights
        weights=    (0.25,0.25,0.25,0.25)):
    if type(ref) is str: ref = [ref]
    reft = []
    for r in ref:
        reft.append(tokenize_words(r))
    cndt = tokenize_words(sen)

    if ngram:
        if ngram == 1: weights = (1,  0,    0,   0)
        if ngram == 2: weights = (0.5,0.5,  0,   0)
        if ngram == 3: weights = (0.33,0.33,0.33,0)

    sb = 0
    try: sb = nltk.translate.bleu_score.sentence_bleu(reft,cndt,weights=weights)
    except ValueError: pass

    return sb

# returns Rouge value/valueL
def rouge_12L(
        sen :str,
        ref :str,
        ev :rouge.Rouge=    None,
        R1=                 True,
        R2=                 True,
        RL=                 True):

    if not ev: ev = rouge.Rouge()
    metrics = [0,0,0]
    try:
        metrics = ev.get_scores(sen,ref)
        metrics = [
            metrics[0]['rouge-1']['f'],
            metrics[0]['rouge-2']['f'],
            metrics[0]['rouge-l']['f']]
    except ValueError: pass
    rm = []
    if R1: rm.append(metrics[0])
    if R2: rm.append(metrics[1])
    if RL: rm.append(metrics[2])
    if len(rm) == 1: rm = rm[0]
    return rm

# selects two furthest (with lev_dist) of three sentences
def two_most_distanced(sa :str,sb :str,sc :str):
    palev = [[lev_dist(*pair), pair] for pair in [[sa, sb], [sb, sc], [sc, sa]]]
    return sorted(palev)[-1][1]


if __name__ == '__main__':

    sen = 'My name is Piotr and I like to work for Samsung today with Majka.'
    ref = 'Her name is Majka and she works for Samsung too.'

    print('lev_dist:  %s'%lev_dist(sen,ref))
    print('Bleu:      %s'%bleu(sen,ref))
    print(lev_distL([0,5,1],[0,1,2,3]))
    print(rouge_12L(sen,ref))
    print(lev_dist_bpe(sen,ref))
