from ._character_to_stroke import Stroke
from ._character_to_radical import Radical
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

    *analyzer* : string, {'char', 'radical', 'stroke'}, default='stroke'

    Whether the feature should be made of character or stroke n-grams.
    """

    def __init__(self, ngram_range=(3, 3), analyzer='stroke'):
        self.analyzer = analyzer
        self.ngram_range = ngram_range

    def _stroke_ngrams(self, word):
        """Tokenize text_document into a sequence of stroke n-grams"""
        min_n, max_n = self.ngram_range

        char_strokes = []
        # bind method outside of loop to reduce overhead
        char_strokes_append = char_strokes.append
        for char in word:
            char_strokes_append(self.tokenizer.get_stroke(char))
        # Separate character with '*'
        word_strokes = '*'.join(char_strokes)
        stroke_len = len(word_strokes)
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the strokes
            ngrams = list(word_strokes)
            min_n += 1
        else:
            ngrams = []
        for n in range(min_n, min(max_n + 1, stroke_len + 1)):
            temp_zip = zip(*[word_strokes[i:] for i in range(n)])
            ngrams += [''.join(ngram) for ngram in temp_zip]
        return ngrams

    def _radical_ngrams(self, word):
        """Tokenize text_document into a sequence of radical n-grams"""
        min_n, max_n = self.ngram_range

        char_radicals = []
        # bind method outside of loop to reduce overhead
        char_radicals_append = char_radicals.append
        for char in word:
            char_radicals_append(self.tokenizer.get_radical(char))
        # Separate character with '*'
        word_radicals = '*'.join(char_radicals)
        radical_len = len(word_radicals)
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the radicals
            ngrams = list(word_radicals)
            min_n += 1
        else:
            ngrams = []
        for n in range(min_n, min(max_n + 1, radical_len + 1)):
            temp_zip = zip(*[word_radicals[i:] for i in range(n)])
            ngrams += [''.join(ngram) for ngram in temp_zip]
        return ngrams

    def _char_ngrams(self, word):
        """Turn word into a sequence of n-grams """
        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_word = word
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original word
                word = list(original_word)
                min_n += 1
            else:
                word = []
            n_original_word = len(original_word)
            for n in range(min_n, min(max_n + 1, n_original_word + 1)):
                temp_zip = zip(*[original_word[i:] for i in range(n)])
                word += [''.join(ngram) for ngram in temp_zip]
        return word

    def _build_analyzer(self):
        if self.analyzer == 'stroke':
            self.tokenizer = Stroke()
            return self._stroke_ngrams
        if self.analyzer == 'radical':
            self.tokenizer = Radical()
            return self._radical_ngrams
        if self.analyzer == 'char':
            return self._char_ngrams
        else:
            default_logger.warning(
                f'Cannot find analyzer \'{self.analyzer}\', use default stroke analyzer instead.'
            )
            return self._stroke_ngrams

    def _validate_data_input(self, X):
        """ The data need to be in one dimension.
        """
        if isinstance(X, pd.DataFrame):
            if X.shape[1] == 1:
                return X.iloc[:, 0].to_numpy()
            else:
                raise Exception('Can only pass 1 dimension data.')
        else:
            X = np.array(X)
            if X.ndim == 1:
                return X
            else:
                raise Exception('Can only pass 1 dimension data.')

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
                if (self.dict_string_list.shape[0] >= n):
                    self.topn_ind_ = np.argpartition(
                        -self.sim_matrix_, range(n), axis=1)[:, :n]
                else:
                    dict_len = self.dict_string_list.shape[0]
                    self.topn_ind_ = np.argpartition(
                        -self.sim_matrix_, range(dict_len),
                        axis=1)[:, :dict_len]
            if (n <= self.topn_ind_.shape[1]):
                return self.dict_string_list[self.topn_ind_[:, :n]]
            else:
                place_filler = np.empty(
                    [self.topn_ind_.shape[0], n - self.topn_ind_.shape[1]])
                place_filler[:] = np.nan
                res = np.append(self.dict_string_list[self.topn_ind_],
                                place_filler, 1)
                return res

    def fit(self, X):
        """
        Learn the target list of the words.

        **Parameters**
        ----------
        *X* : list, pd.Series, 1d np.array or 1d pd.DataFrame

        an iterable yields chinese str in utf-8

        **Returns**
        -------
        *self* : FuzzyChinese object
        """

        self.dict_string_list = self._validate_data_input(X)
        if isinstance(X, pd.Series) | isinstance(X, pd.DataFrame):
            self._X_index = X.index.to_numpy()
        self.dict_feature_matrix_ = self._vectorize_dict(self.dict_string_list)
        return self

    def fit_transform(self, X, Y=None, n=3):
        """
        Learn the words in X.
        If Y is not passed, then find similar words in the X itself .
        If Y is passed, for each word in Y, find the similar words in X.

        **Parameters**
        ----------

        *X* : list, pd.Series, 1d np.array or 1d pd.DataFrame

        *Y* : list, pd.Series, 1d np.array or 1d pd.DataFrame

        X and Y are iterables yield chinese str in utf-8
            
        *n* : int

        top n matched to be returned

        **Returns**
        -------
        *res* : A numpy matrix. [n_samples, n_matches]

        Each row corresponds to the top n matches to the input row. 
        Matches are sorted by descending order in similarity.
        """
        if isinstance(X, pd.Series) | isinstance(X, pd.DataFrame):
            self._X_index = X.index.to_numpy()
        X = self._validate_data_input(X)
        self.dict_string_list = X
        if Y is not None:
            Y = self._validate_data_input(Y)
            self.Y_string_list = Y
            feature_matrix_ = self._vectorize_dict(np.append(X, Y))
            self.dict_feature_matrix_ = feature_matrix_[:len(X)]
            self.Y_feature_matrix_ = feature_matrix_[len(X):]
            self._get_cosine_similarity()
        else:
            if (~hasattr(self, 'dict_string_list')
                    or self.dict_string_list != X):
                self.fit(X)
                self.Y_feature_matrix_ = self.dict_feature_matrix_
                self._get_cosine_similarity()
        return self._get_top_n_similar(n)

    def transform(self, Y, n=3):
        """
        Match the list of words to a target list of words.

        **Parameters**
        ----------
        *Y* : list, pd.Series, 1d np.array or 1d pd.DataFrame

        an iterable yields chinese str in utf-8

        *n* : int

        top n matched to be returned

        **Returns**
        -------
        *res* : A numpy matrix. [n_samples, n_matches]

        Each row corresponds to the top n matches to the input row. 
        Matches are sorted by descending order in similarity.
        """
        Y = self._validate_data_input(Y)
        if (~hasattr(self, 'Y_string_list') or self.Y_string_list != Y):
            self.Y_string_list = Y
            self.Y_feature_matrix_ = self._vectorize_Y(Y)
            self._get_cosine_similarity()
        return self._get_top_n_similar(n)

    def get_similarity_score(self):
        """
        Return the similarity score for last transform call.

        **Returns**
        -------

        *res* : A numpy matrix. 
        
        Each row corresponds to the similarity score of 
        top n matches.
        """

        if hasattr(self, 'Y_feature_matrix_'):
            return np.take_along_axis(self.sim_matrix_, self.topn_ind_, axis=1)
        else:
            raise Exception('Must run transform or fit_transform first.')

    def get_index(self):
        """
        Return the original index of the matched word.

        **Returns**
        -------

        *res* : A numpy matrix. 
        
        Each row corresponds to the index of 
        top n matches. Original index is return if exists.
        """

        if hasattr(self, 'topn_ind_'):
            if hasattr(self, '_X_index'):
                return self._X_index[self.topn_ind_]
            else:
                return self.topn_ind_
        else:
            raise Exception('Must run transform or fit_transform first.')

    def compare_two_columns(self, X, Y):
        """
        Compare two columns and calculated similarity score for each pair on each row.

        **Parameters**
        ----------
        *X* : list, pd.Series, 1d np.array or 1d pd.DataFrame
        
        *Y* : list, pd.Series, 1d np.array or 1d pd.DataFrame, have same length as X

        **Returns**
        -------

        *res* : A numpy matrix. 
        
        Return two original columns and a new column for the similarity score.
        """
        X = self._validate_data_input(X)
        Y = self._validate_data_input(Y)
        if len(X) != len(Y):
            raise Exception('The columns passed have different length!')
        feature_matrix_ = self._vectorize_dict(np.append(X, Y))
        X_feature_matrix_ = feature_matrix_[:len(X)]
        Y_feature_matrix_ = feature_matrix_[len(X):]

        # Perform rowwise dot product
        similarity_score = np.einsum('ij,ij->i', X_feature_matrix_.toarray(),
                                     Y_feature_matrix_.toarray())
        return np.array([X, Y, similarity_score]).T

    def __repr__(self):
        return f'FuzzyChineseMatch(analyzer={self.analyzer}, ngram_range={self.ngram_range})'


if __name__ == "__main__":
    test_dict = pd.Series(
        ['长白朝鲜族自治县', '长阳土家族自治县', '城步苗族自治县', '达尔罕茂明安联合旗', '汨罗市'])
    raw_word = pd.Series(['达茂联合旗', '长阳县', '汩罗市'])
    assert ('汩罗市' != '汨罗市')  # They are not the same!

    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='stroke')
    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='radical')
    fcm = FuzzyChineseMatch(ngram_range=(3, 3), analyzer='char')
    fcm.fit(test_dict)
    top2_similar = fcm.transform(raw_word, n=2)
    res = pd.concat([
        raw_word,
        pd.DataFrame(top2_similar, columns=['top1', 'top2']),
        pd.DataFrame(
            fcm.get_similarity_score(), columns=['top1_score', 'top2_score']),
        pd.DataFrame(fcm.get_index(), columns=['top1_index', 'top2_index'])
    ],
                    axis=1)
    fcm.compare_two_columns(res.town_name, res.top1)
