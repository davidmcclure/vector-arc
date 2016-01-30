

import regex


class Text:


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

        pattern = regex.finditer('\p{L}', self.text.lower())

        # TODO: stem?

        for match in pattern:
            self.tokens.append(match.group(0))
