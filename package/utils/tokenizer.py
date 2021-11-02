from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict, defaultdict
import string
import re
def text_to_word_sequence(text, sep=" ", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True):
    seq = []
    if lower:
        text = text.lower()

    re_filters = re.compile(re.escape("[ %s ]" % filters))
    text = re.sub(pattern=re_filters, repl=sep, string=text)
    for txt in text.split(sep):
        if txt is None:
            continue
        seq.append(txt)
    return seq

class Tokenizer(object):
    def __init__(self, num_words=None,
                 sep=" ",
                 filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 lower=True,
                 char_level=False,
                 oov_token=None,
                 **kwargs):

        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        self.num_words = num_words

        self.word_counts = OrderedDict()
        self.word_doc = defaultdict(int)

        self.sep = sep
        self.filters = filters
        self.lower = lower
        self.char_level = char_level
        self.oov_token = oov_token
        self.doc_count = 0

        self.word2idx = dict()
        self.idx2word = dict()

        self.ind_doc = dict()

    def fit_on_texts(self, texts):
        assert isinstance(texts, list)

        for text in texts:
            self.doc_count += 1
            if isinstance(texts, list) or self.char_level:
                if self.lower:
                    if isinstance(text, list):
                        text = [txt.strip().lower() for txt in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text, self.sep, self.filters, self.lower)

            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1

            for w in set(seq):
                self.word_doc[w] += 1

        word_order = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)

        if self.oov_token is None:
            vocab = []
        else:
            vocab = [self.oov_token]

        vocab.extend([w[0] for w in word_order])
        index = list(range(1, len(vocab)+1))

        self.word2idx = dict((word, ind) for word, ind in list(zip(vocab, index)))
        self.idx2word = dict((ind, word) for ind, word in list(zip(index, vocab)))

        for w, c in self.word_doc.items():
            self.ind_doc[self.word2idx[w]] = c

    def texts_to_sequences(self, texts):
        num_words = self.num_words
        seq_return = []
        for text in texts:
            if isinstance(texts, list) or self.char_level:
                if self.lower:
                    if isinstance(text, list):
                        text = [txt.strip().lower() for txt in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                seq = text_to_word_sequence(text, self.sep, self.filters, self.lower)

            seq_element = []
            i = 0
            index_oov_token = self.word2idx.get(self.oov_token)
            for w in seq:
                i = self.word2idx.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if index_oov_token is not None:
                            seq_element.append(self.idx2word.get(index_oov_token))
                    else:
                        seq_element.append(i)
                elif self.oov_token is not None:
                    seq_element.append(self.idx2word.get(index_oov_token))
            seq_return.append(seq_element)
        return seq_return
