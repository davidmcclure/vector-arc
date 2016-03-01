

import regex
import numpy as np

from gensim.models import Word2Vec
from itertools import islice


# train a word2vec model
# given a set of words, compute a "breadth" metric
# slide window across text


def window(seq, n=2):

    """
    Yield a sliding window over an iterable.

    Args:
        seq (iter): The sequence.
        n (int): The window width.

    Yields:
        tuple: The next window.
    """

    it = iter(seq)
    result = tuple(islice(it, n))

    if len(result) == n:
        yield result

    for token in it:
        result = result[1:] + (token,)
        yield result


class Text:

    @classmethod
    def from_file(cls, path):

        """
        Hydrate a text from a file.

        Args:
            path (str)
        """

        with open(path, 'r') as fh:
            return cls(fh.read())

    def __init__(self, text):

        """
        Set the raw text string, tokenize.

        Args:
            text (str)
        """

        self.text = text

        self.tokenize()

    def tokenize(self):

        """
        Tokenize the text.
        """

        self.tokens = []

        pattern = regex.finditer('\p{L}+', self.text.lower())

        # TODO: stem?

        for match in pattern:
            self.tokens.append(match.group(0))

    def mean_norm_series(self, model, n=1000):

        """
        Get the token mean norms for a sliding window.

        Args:
            model (Model)
            n (int) - Window width.

        Returns: list
        """

        series = []

        for w in window(self.tokens, n):
            series.append(model.mean_norm(w))

        return series


class Model(Word2Vec):

    def mean_norm(self, tokens):

        """
        Compute the norm of the average vector for a set of tokens.

        Args:
            tokens (list)

        Returns: float
        """

        vectors = []
        for t in tokens:
            if t in self:
                vectors.append(self[t])

        mean = sum(vectors) / len(vectors)

        return np.linalg.norm(mean)
