"""

 2018 (c) piteren

"""

import string


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
    text_list = whitspace_tokenizer(text)
    return ' '.join(text_list)


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