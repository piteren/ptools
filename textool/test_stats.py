"""

 2021 (c) piteren

"""

from ptools.plots import histogram


def docs_stats(
        docs: list,
        print_pd=       True,   # to print stats with pandas
        save_FD :str=   None):  # to save plots to folder


    print(f'\nStats for {len(docs)} documents:')

    vals = {
        'n_sen_doc':    [], # num sentences  per doc
        'n_wrd_doc':    [], # num words      per doc
        'n_chr_doc':    [], # num chars      per doc
        'n_wrd_sen':    [], # num words      per sentence
        'n_chr_sen':    [], # num chars      per sentence
        'n_chr_wrd':    []} # num chars      per word

    for doc in docs:
        doc_words = 0
        doc_chars = 0
        for sen in doc:
            n_chars = len(sen)
            doc_chars += n_chars
            vals['n_chr_sen'].append(n_chars)

            words = sen.split()
            n_words = len(words)
            doc_words += n_words
            vals['n_wrd_sen'].append(n_words)
            for word in words:
                vals['n_chr_wrd'].append(len(word))

        vals['n_sen_doc'].append(len(doc))
        vals['n_wrd_doc'].append(doc_words)
        vals['n_chr_doc'].append(doc_chars)

    for k in vals:
        print(f'stats for {k}:')
        histogram(
            val_list=   vals[k],
            name=       k,
            print_pd=   print_pd,
            save_FD=    save_FD)


if __name__ == '__main__':

    docA = [
        'To jest przykładowe zdanie.',
        'Inne zdanie, inne od pierwszego.',
        'Kolejne zdanie, które jest w dokumencie.']
    docB = [
        'W dokumencie B są zupełnie inne zdania.',
        'Trudno je nawet zliczyć.',
        'Ale można do tego użyć narzędzi statystycznych.',
        'Mozna to zrobić w python\'ie.']
    docC = [
        'W tym dokumencie jest tylko jedno zdanie, ale za to najdłuższe']

    docs = [docA,docB,docC]

    docs_stats(docs)