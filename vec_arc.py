

import regex
import numpy as np

from gensim.models import Word2Vec
from itertools import islice, combinations, chain
from clint.textui import progress
from stop_words import get_stop_words


def window(seq, n):

    """
    Yield a sliding window over an iterable.

    Args:
        seq (iter): The sequence.
        n (int): Window width.

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


def chunks(seq, n):

    """
    Yield "groups" from an iterable.

    Args:
        seq (iter): The iterable.
        n (int): Chunk size.

    Yields:
        The next chunk.
    """

    for i in range(0, len(seq), n):
        yield seq[i:i+n]


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

        stopwords = get_stop_words('en')

        # TODO: stem?

        for match in pattern:

            token = match.group(0)

            # Exclude stop words.
            if token not in stopwords:
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

    def mean_cosine_chunks(self, model, n=1000):

        """
        Get the mean pairwise similarities for token chunks

        Args:
            model (Model)
            n (int) - Chunk size.

        Returns: list
        """

        series = []

        for i, w in enumerate(chunks(self.tokens, n)):
            series.append(model.mean_cosine(w))
            print(i)

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

    def mean_cosine(self, tokens):

        """
        Compute the average cosine similarity between all pairs of tokens.

        Args:
            tokens (list)

        Returns: float
        """

        distances = []

        for t1, t2 in combinations(tokens, 2):
            if t1 in self and t2 in self:
                distances.append(self.similarity(t1, t2))

        return np.mean(distances)
