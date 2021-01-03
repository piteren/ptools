"""

 2020 (c) piteren

"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def fit_docs(
        docs :list,
        vectorizer=     None, # for given vectorizer uses its vocab and idf
        ngram_range=    (1,1),
        tfidf_feats=    None,
        vocab=          None,
        verb=           0):

    if not vectorizer:
        # build vectorizer and fit
        vectorizer = TfidfVectorizer(
            use_idf=        True,
            ngram_range=    ngram_range,
            max_features=   tfidf_feats,
            vocabulary=     vocab,
            stop_words=     'english')
        vectorizer.fit(docs)

    tfidf = vectorizer.transform(docs)
    if verb > 0:
        tf_shape = tfidf.shape
        print(f'Prepared TFIDF for {tf_shape[0]} documents with {tf_shape[1]} vocab')

    return {
        'vectorizer':   vectorizer,
        'vocab':        vectorizer.get_feature_names(),
        'idf':          vectorizer.idf_,
        'tfidf_sparse': tfidf,
        'tfidf':        [np.squeeze(f.toarray()) for f in tfidf]}