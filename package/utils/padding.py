from copy import deepcopy
import warnings
from collections.abc import Iterable


class Padding(object):
    def __init__(self, max_length=None,
                 replace_by=0,
                 padding="pre",
                 **kwargs):

        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        if padding not in ["pre", "post", "center"]:
            raise ValueError("padding must be pre, post or center")

        self.replace_by = replace_by if replace_by else 0
        self.padding = padding
        self.max_length = max_length

    def _padding_str(self, texts):
        max_length = len(texts)
        text_copy = deepcopy(texts)
        if self.max_length is None or max_length >= self.max_length:
            self.max_length = max_length
            return text_copy
        else:
            if self.padding == "pre":
                while len(text_copy) < self.max_length:
                    text_copy = str(self.replace_by) + text_copy
            elif self.padding == "post":
                while len(text_copy) < self.max_length:
                    text_copy = text_copy + str(self.replace_by)
            else:
                while len(text_copy) < self.max_length:
                    text_copy = text_copy + str(self.replace_by)
                    if len(text_copy) >= self.max_length:
                        break
                    else:
                        text_copy = str(self.replace_by) + text_copy

        return text_copy

    def _padding_list(self, texts):
        max_length = max(len(text) for text in texts)
        text_copy = deepcopy(texts)
        if self.max_length is None or max_length >= self.max_length:
            self.max_length = max_length

        new_text = []
        for text in text_copy:
            if self.padding == "pre":
                while len(text) < self.max_length:
                    if isinstance(text, str):
                        text = str(self.replace_by) + text
                    elif isinstance(text, Iterable):
                        text = list(text)
                        text = text + [self.replace_by]

            elif self.padding == "post":
                while len(text) < self.max_length:
                    if isinstance(text, str):
                        text = text + str(self.replace_by)
                    elif isinstance(text, Iterable):
                        text = list(text)
                        text = [self.replace_by] + text
            else:
                while len(text) < self.max_length:
                    if isinstance(text, str):
                        text = text + str(self.replace_by)
                        if len(text) >= self.max_length:
                            break
                        else:
                            text = str(self.replace_by) + text
                    elif isinstance(text, Iterable):
                        text = list(text)
                        text = text + [self.replace_by]
                        if len(text) >= self.max_length:
                            break
                        else:
                            text = [self.replace_by] + text
            new_text.append(text)
        return new_text

    def _padding(self, texts):
        if isinstance(texts, str):
            return self._padding_str(texts)
        if isinstance(texts, Iterable):
            return self._padding_list(texts)
        raise TypeError("texts must be type str or iterable")

    def __call__(self, texts):
        return self._padding(texts)

    def __str__(self):
        return "padding"
