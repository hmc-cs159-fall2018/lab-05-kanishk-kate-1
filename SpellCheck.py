import spacy
import EditDistance
from LanguageModel import LanguageModel
import string
import argparse


nlp = spacy.load("en", pipeline=["tagger", "parser"])


class SpellChecker:
    def __init__(self, max_distance, channel_model=None, language_model=None, threshold=15):
        self.nlp = nlp
        self.channel_model = channel_model
        self.language_model = language_model
        self.max_distance = max_distance
        self.unknown_words = dict()
        self.threshold = threshold

    def load_channel_model(self, fp):
        self.channel_model = EditDistance.EditDistanceFinder()
        self.channel_model.load(fp)

    def load_language_model(self, fp):
        self.language_model = LanguageModel()
        self.language_model.load(fp)

    def bigram_score(self, prev_word, focus_word, next_word):
        bigram_prob1 = self.language_model.bigram_prob(prev_word, focus_word)
        bigram_prob2 = self.language_model.bigram_prob(focus_word, next_word)
        return (bigram_prob1 + bigram_prob2) / 2

    def unigram_score(self, word):
        return self.language_model.unigram_prob(word)

    def cm_score(self, error_word, corrected_word):
        return self.channel_model.prob(error_word, corrected_word)

    def inserts(self, word):
        one_insert_away = []
        alphabet = string.ascii_lowercase
        for i in range(len(word) + 1):
            for letter in alphabet:
                new_word = word[0:i] + letter + word[i:]
                if new_word in self.language_model:
                    one_insert_away.append(new_word)
        return one_insert_away

    def deletes(self, word):
        one_delete_away = []
        for i in range(len(word)):
            new_word = word[0:i] + word[i + 1:]
            if new_word in self.language_model:
                one_delete_away.append(new_word)
        return one_delete_away

    def substitutes(self, word):
        one_sub_away = []
        alphabet = string.ascii_lowercase
        for i in range(len(word)):
            for letter in alphabet:
                if letter != word[i]:
                    new_word = word[0:i] + letter + word[i + 1:]
                    if new_word in self.language_model:
                        one_sub_away.append(new_word)
        return one_sub_away

    def generate_candidates_recurse(self, word_list, max_distance):
        if max_distance == 0:
            return word_list
        new_list = []
        for i in word_list:
            insert_words = self.inserts(i)
            delete_words = self.deletes(i)
            sub_words = self.substitutes(i)
            new_list += insert_words
            new_list += delete_words
            new_list += sub_words
        set_list = list(set(new_list))
        return self.generate_candidates_recurse(set_list, max_distance - 1)

    def generate_candidates(self, word):
        return self.generate_candidates_recurse([word], self.max_distance)

    def check_sentence(self, sentence, fallback=False):
        return_list = []
        for i in sentence:
            if i in self.language_model:
                return_list.append([i])
                continue
            if i in self.unknown_words:
                if self.unknown_words[i] > self.threshold:
                    return_list.append([i])
                    continue
                self.unknown_words[i] += 1
            elif i not in self.unknown_words:
                self.unknown_words[i] = 1
            candidates = self.generate_candidates(i)
            if candidates == []:
                if fallback:
                    return_list.append([i])
                    continue
                else:
                    return_list.append([])
                    continue
            candidates = sorted(candidates, key = lambda x: self.unigram_score(x) + self.cm_score(i, x), reverse = True)
            return_list.append(candidates)
        return return_list

    def check_line(self, text, fallback=False):
        sentence_doc = nlp(text)
        sentences = sentence_doc.sents
        sentences = [self.language_model.get_tokens(sentence) for sentence in sentences]
        result = []
        for sentence in sentences:
            checked_sentence = self.check_sentence(sentence)
            result += checked_sentence
        return result

    def autocorrect_sentence(self, sentence):
        possibilities = self.check_sentence(sentence, True)
        possibilities = [x[0] for x in possibilities]
        return possibilities

    def autocorrect_line(self, line):
        doc = nlp(line)
        sentences = doc.sents
        sentences = [self.language_model.get_tokens(sentence) for sentence in sentences]
        result = []
        for sentence in sentences:
            checked_sentence = self.autocorrect_sentence(sentence)
            result += checked_sentence
        return result

    def suggest_sentence(self, sentence, max_suggestions):
        possibilities = self.check_sentence(sentence, True)
        return_list = []
        for i in possibilities:
            if len(i) == 1:
                return_list.append(i[0])
            else:
                return_list.append(i[:max_suggestions])
        return return_list

    def suggest_line(self, text, max_suggestions):
        doc = nlp(text)
        sentences = doc.sents
        sentences = [self.language_model.get_tokens(sentence) for sentence in sentences]
        result = []
        for sentence in sentences:
            checked_sentence = self.suggest_sentence(sentence, max_suggestions)
            result += checked_sentence
        return result
