"""

 2020 (c) piteren

"""

from sklearn.feature_extraction.text import TfidfVectorizer

# prepares tfidf for documents list
def fit_docs(
        docs :list,
        vectorizer=     None, # for given vectorizer uses its vocab and idf
        max_features=   None,
        vocab=          None,
        verb=           0):

    if not vectorizer:
        # build vectorizer and fit_transform
        vectorizer = TfidfVectorizer(
            use_idf=        True,
            max_features=   max_features,
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
        'tfidf':        tfidf}
