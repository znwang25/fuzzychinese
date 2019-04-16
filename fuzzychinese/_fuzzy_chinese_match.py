from ._character_to_stroke import Stroke
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import logging
default_logger = logging.getLogger(__name__)


class FuzzyChineseMatch(object):
    """ 
    Match a collection of chinese words with a target list of words.

    **Parameters**
    ----------
    *ngram_range* : tuple (min_n, max_n), default=(3, 3)

    The lower and upper boundary of the range of n-values for different
    n-grams to be extracted. All values of n such that min_n <= n <= max_n
    will be used.

    *analyzer* : string, {'char', 'stroke'}

    Whether the feature should be made of character or stroke n-grams.
    """

    def __init__(self, ngram_range=(3, 3), analyzer='stroke'):
        self.analyzer = analyzer
        self.ngram_range = ngram_range

    def _stroke_ngrams(self, string):
        """Tokenize text_document into a sequence of character n-grams"""
        min_n, max_n = self.ngram_range

        char_strokes = []
        # bind method outside of loop to reduce overhead
        char_strokes_append = char_strokes.append
        for char in string:
            char_strokes_append(self.stroke_op.get_stroke(char))
        # Separate character with '*'
        string_strokes = '*'.join(char_strokes)
        stroke_len = len(string_strokes)
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the strokes
            ngrams = list(string_strokes)
            min_n += 1
        else:
            ngrams = []
        for n in range(min_n, min(max_n + 1, stroke_len + 1)):
            temp_zip = zip(*[string_strokes[i:] for i in range(n)])
            ngrams += [''.join(ngram) for ngram in temp_zip]
        return ngrams

    def _char_ngrams(self, string):
        """Turn string into a sequence of n-grams """
        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_string = string
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original string
                string = list(original_string)
                min_n += 1
            else:
                string = []
            n_original_string = len(original_string)
            for n in range(min_n, min(max_n + 1, n_original_string + 1)):
                temp_zip = zip(*[original_string[i:] for i in range(n)])
                string += [''.join(ngram) for ngram in temp_zip]
        return string

    def _build_analyzer(self):
        if self.analyzer == 'stroke':
            self.stroke_op = Stroke()
            return self._stroke_ngrams

        if self.analyzer == 'char':
            return self._char_ngrams

    def _vectorize_dict(self, raw_documents):
        """ Vectorize the dictionary documents.
            Create sparse feature matrix, and vocabulary.
        """
        analyzer = self._build_analyzer()
        default_logger.debug('Vectorizing dictionary documents ...')
        self._vectorizer = TfidfVectorizer(
            min_df=1, analyzer=analyzer, norm='l2')
        X = self._vectorizer.fit_transform(raw_documents)
        self.idf_ = self._vectorizer.idf_
        self.vocabulary_ = self._vectorizer.vocabulary_
        return X

    def _vectorize_Y(self, raw_documents):
        """ Vectorize documents need to be matched.
            Create sparse feature matrix, and vocabulary.
        """
        default_logger.debug('Vectorizing documents to be matched ...')
        Y = self._vectorizer.transform(raw_documents)
        return Y

    def _get_cosine_similarity(self):
        """ Calculate cosine similarity.
        """
        default_logger.debug('Calculating cosine similarity ...')
        if hasattr(self, 'dict_feature_matrix_'):
            self.sim_matrix_ = self.Y_feature_matrix_.dot(
                self.dict_feature_matrix_.T).toarray()
        else:
            raise Exception('Need to fit dictionary first.')

    def _get_top_n_similar(self, n):
        """ Find the top n similar words from cosine similarity matrix.
        """
        default_logger.debug('Finding the top n similar words ...')
        if hasattr(self, 'sim_matrix_'):
            if ~hasattr(self, 'topn_ind_') or (self.topn_ind_.shape[1] < n):
                self.topn_ind_ = np.argpartition(
                    -self.sim_matrix_, range(n), axis=1)[:, :n]
            if (self.topn_ind_.shape[1] > n):
                return list(
                    map(
                        lambda x: self.dict_string_list[x],
                        self.topn_ind_[:, :n],
                    ))
            else:
                return list(
                    map(
                        lambda x: self.dict_string_list[x],
                        self.topn_ind_,
                    ))

    def fit(self, X):
        """
        Learn the target list of the words.

        **Parameters**
        ----------
        *X* : iterable

        an iterable which yields str

        **Returns**
        -------
        *self* : FuzzyChinese object
        """

        self.dict_feature_matrix_ = self._vectorize_dict(X)
        self.dict_string_list = np.array(X)
        return self

    def fit_transform(self, X, n=3):
        """
        Learn the target list of the words.
        Find similar words in the target itself.

        **Parameters**
        ----------
        *Y* : iterable

        an iterable which yields str
            
        *n* : int

        top n matched to be returned

        **Returns**
        -------
        *X* : A numpy matrix. [n_samples, n_matches]

        Each row corresponds to the top n matches to the input row. 
        Matches are sorted by descending order in similarity.
        """

        if (~hasattr(self, 'dict_string_list') or self.dict_string_list != X):
            self.fit(X)
            self.Y_feature_matrix_ = self.dict_feature_matrix_
            self._get_cosine_similarity()
        return self._get_top_n_similar(n)

    def transform(self, Y, n=3):
        """
        Match the list of words to a target list of words.

        **Parameters**
        ----------
        *Y* : iterable

        an iterable which yields either str

        *n* : int

        top n matched to be returned

        **Returns**
        -------
        *X* : A numpy matrix. [n_samples, n_matches]

        Each row corresponds to the top n matches to the input row. 
        Matches are sorted by descending order in similarity.
        """
        self.Y_string_list = np.array(Y)
        if (~hasattr(self, 'Y_string_list') or self.Y_string_list != Y):
            self.Y_feature_matrix_ = self._vectorize_Y(Y)
            self._get_cosine_similarity()
        return self._get_top_n_similar(n)

    def get_similarity_score(self):
        """
        Return the similarity score for last transform call.

        **Returns**
        -------

        *X* : A numpy matrix. 
        
        Each row corresponds to the similarity score of 
        top n matches.
        """

        if hasattr(self, 'Y_feature_matrix_'):
            return np.take_along_axis(self.sim_matrix_, self.topn_ind_, axis=1)
        else:
            raise Exception('Must run transform or fit_transform first.')

    def __repr__(self):
        return f'FuzzyChineseMatch(analyzer={self.analyzer}, ngram_range={self.ngram_range})'


if __name__ == "__main__":
    dict_list = pd.read_csv('test/townname_dict.csv').chname
    source_list = pd.read_csv('test/townname_raw.csv').town_name
    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
    fcm.fit(dict_list)
    top3_similar = fcm.transform(source_list[:500], n=3)
    top3_similar = fcm.fit_transform(source_list[:500], n=3)
    res = pd.concat([
        source_list[0:500],
        pd.DataFrame(top3_similar, columns=['top1', 'top2', 'top3']),
        pd.DataFrame(
            fcm.get_similarity_score(),
            columns=['top1_score', 'top2_score', 'top3_score'])
    ],
                    axis=1)
