"""

 2018 (c) piteren

"""

import nltk
import spacy
import string

SPACY_EN_NLP = spacy.load('en')

# separates punctuation with spaces
def pretokenize_punct(text):
    newText = ''
    for ix in range(len(text)):
        c = text[ix]
        if c in string.punctuation:
            newText += ' %s '%c
        else: newText += c
    return newText

# whitespace tokenizer
def whitspace_tokenizer(text):
    return text.split()

# whitespace tokenizer
def whitspace_normalize(text):
    text_list = text.split()
    return ' '.join(text_list)

# spacy word tokenizer
def spacy_Wtokenizer(text):
    tokens = SPACY_EN_NLP.tokenizer(text)
    return [w.text for w in tokens]

# spacy sentence tokenizer
def spacy_Stokenizer(text):
    tokens = SPACY_EN_NLP(text).sents
    return [s.text for s in tokens]

# word tokenization method, supports 'spacy', 'nltk', for other 'space'
def tokenize_words(
        text,
        tokenizer=  'spacy'):

    if tokenizer == 'spacy':    return spacy_Wtokenizer(text)
    if tokenizer == 'nltk':     return nltk.word_tokenize(text)
    return whitspace_tokenizer(text)

# sentence tokenization method
def tokenize_sentences(
        text,
        tokenizer=  'spacy'):

    assert tokenizer in ['spacy','nltk']
    if tokenizer == 'spacy': return spacy_Stokenizer(text)
    return nltk.sent_tokenize(text)


if __name__ == '__main__':

    text = 'Just   how is  Hillary      Kerr, the founder of a digital media company in Los Angeles? She can tell you what song was playing five years ago on the jukebox at the bar where she somewhat randomly met the man who became her husband.'

    print('>%s<'%whitspace_normalize(text))

    """
    print(pretokenize_punct(text))
    print('\n***words:')
    for word in tokenize_words(text, tokenizer=None): print(' >%s<' % word)
    print('\n***sentences:')
    for sent in tokenize_sentences(text): print(' >%s<' % sent)
    """