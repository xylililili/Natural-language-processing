# import random
# from collections import defaultdict
# from typing import List
# import nltk


# class BackoffNGramModel:
#     def __init__(self, corpus, n):
#         self.ngram_model = self.build_ngram_model(corpus, n)
#         self.n = n
#         self.corpus = corpus
#         # self.most_frequent_word = self.get_most_frequentword(corpus)

#     def get_most_frequentword(self, corpus):
#         word_freq = defaultdict(int)
#         most_frequent = 0
#         ans = ""
#         for word in corpus:
#             word_freq[word] += 1
#             if word_freq[word] > most_frequent:
#                 most_frequent = word_freq[word]
#                 ans = word
#         return ans

#     def build_ngram_model(self, corpus, n):
#         """Build an n-gram model from a source corpus."""
#         # use a dic to record the corresponding next tokens of the current gram
#         ngram_model = defaultdict(list)
#         for i in range(len(corpus) - n):
#             ngram = tuple(corpus[i:i + n-1])
#             # print("ngram:", ngram)
#             next_word = corpus[i + n-1]
#             # print("next_word:", next_word)
#             ngram_model[ngram].append(next_word)
#         return ngram_model

#     def generate_next_word(self, current_ngram, corpus, randomize=False):
#         if current_ngram in self.ngram_model:
#             next_word_options = self.ngram_model[current_ngram]
#             # if random is true, randomlly choose the words that occured after the current words
#             if randomize:
#                 return random.choice(next_word_options)
#             # choose the words that occured most frequently
#             else:
#                 return max(set(next_word_options), key=next_word_options.count)
#         else:
#             # Backoff: If n-gram is unseen, back off to (n-1)-gram
#             n_minus_1_gram = current_ngram[:-1]
#             if len(current_ngram) == 1:
#                 # print(corpus)
#                 # return ","
#                 return self.get_most_frequentword(corpus)
#             if n_minus_1_gram:
#                 # generate the n-1 gram model, if can't find current gram
#                 n_minus_1_model = BackoffNGramModel(self.corpus, self.n - 1)

#                 return n_minus_1_model.generate_next_word(n_minus_1_gram, corpus, randomize)
#             # else:
#                 # If we reach unigram level and it does not exist in the corpus


# def finish_sentence(sentence, n, corpus, randomize=False):
#     """Extend a sentence using an n-gram model until a stopping condition is met."""
#     if n < 1:
#         raise ValueError("n must greater 1")

#     extended_sentence = sentence.copy()
#     model = BackoffNGramModel(corpus, n)

#     while len(extended_sentence) < 10:
#         current_ngram = tuple(
#             extended_sentence[-n+1:]) if (len(extended_sentence) >= n) else tuple(extended_sentence)
#         # print(current_ngram)
#         next_word = model.generate_next_word(current_ngram, corpus, randomize)
#         # print(next_word)
#         if next_word is None:
#             # extended_sentence.append(',')
#             break
#         else:
#             extended_sentence.append(next_word)
#         if next_word in (".", "?", "!"):
#             break

#     return extended_sentence
import random
from collections import defaultdict
from typing import List
import nltk


class BackoffNGramModel:
    def __init__(self, corpus, n):
        self.ngram_model = self.build_ngram_model(corpus, n)
        self.n = n
        self.corpus = corpus
        # self.most_frequent_word = self.get_most_frequentword(corpus)

    def get_most_frequentword(self, corpus):
        word_freq = defaultdict(int)
        most_frequent = 0
        ans = ""
        for word in corpus:
            word_freq[word] += 1
            if word_freq[word] > most_frequent:
                most_frequent = word_freq[word]
                ans = word
        return ans

    def build_ngram_model(self, corpus, n):
        """Build an n-gram model from a source corpus."""
        # use a dic to record the corresponding next tokens of the current gram
        ngram_model = defaultdict(list)
        for i in range(len(corpus) - n):
            ngram = tuple(corpus[i:i + n-1])
            # print("ngram:", ngram)
            next_word = corpus[i + n-1]
            # print("next_word:", next_word)
            ngram_model[ngram].append(next_word)
        return ngram_model

    def generate_next_word(self, current_ngram, corpus, randomize=False):
        if current_ngram in self.ngram_model:
            next_word_options = self.ngram_model[current_ngram]
            # if random is true, randomlly choose the words that occured after the current words
            if randomize:
                return random.choice(next_word_options)
            # choose the words that occured most frequently
            else:
                return max(set(next_word_options), key=next_word_options.count)
        else:
            # Backoff: If n-gram is unseen, back off to (n-1)-gram
            n_minus_1_gram = current_ngram[:-1]
            if len(current_ngram) == 1:
                # print(corpus)
                # return ","
                return self.get_most_frequentword(corpus)
            if n_minus_1_gram:
                # generate the n-1 gram model, if can't find current gram
                n_minus_1_model = BackoffNGramModel(self.corpus, self.n - 1)

                return n_minus_1_model.generate_next_word(n_minus_1_gram, corpus, randomize)
            # else:
                # If we reach unigram level and it does not exist in the corpus


def finish_sentence(sentence, n, corpus, randomize=False):
    """Extend a sentence using an n-gram model until a stopping condition is met."""
    if n < 1:
        raise ValueError("n must greater 1")

    extended_sentence = sentence.copy()
    model = BackoffNGramModel(corpus, n)

    while len(extended_sentence) < 10:
        current_ngram = tuple(
            extended_sentence[-n+1:]) if (len(extended_sentence) >= n) else tuple(extended_sentence)
        # print(current_ngram)
        next_word = model.generate_next_word(current_ngram, corpus, randomize)
        # print(next_word)
        if next_word is None:
            # extended_sentence.append(',')
            break
        else:
            extended_sentence.append(next_word)
        if next_word in (".", "?", "!"):
            break

    return extended_sentence
