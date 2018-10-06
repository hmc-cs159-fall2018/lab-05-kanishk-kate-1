import spacy
import EditDistanceFinder
from LanguageModel import LanguageModel


nlp = spacy.load("en", pipeline=["tagger", "parser"])


class SpellChecker:
    def __init__(self, channel_model=None, language_model=None, max_distance):
        self.nlp = nlp
        self.channel_model = channel_model
        self.language_model = language_model
        self.max_distance = max_distance

    def load_channel_model(self, fp):
        self.channel_model = EditDistanceFinder()
        self.load(fp)

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

    def inserts(word):
        one_insert_away = []
        for model_word in self.language_model.vocabulary:
            _, alignment = self.channel_model.align(word, new_word)
            # If the length is too long, this word doesn't work
            if len(alignment) > len(word) + 1:
                break
            num_inserts = 0
            for pair in alignment:
                if pair[0] != pair[1]:
                    if pair[0] == '%' and num_inserts == 0:
                        num_inserts += 1
                    else:
                        num_inserts = -1
                        break
                if num_inserts == 1:
                    one_insert_away.append(model_word)
        return one_insert_away

    def deletes(word):
        one_delete_away = []
        for model_word in self.language_model.vocabulary:
            _, alignment = self.channel_model.align(word, new_word)
            # If the length is too long, this word doesn't work
            if len(alignment) != len(word):
                break
            num_deletes = 0
            for pair in alignment:
                if pair[0] != pair[1]:
                    if pair[1] == '%' and num_deletes == 0:
                        num_deletes += 1
                    else:
                        num_deletes = -1
                        break
                if num_deletes == 1:
                    one_delete_away.append(model_word)
        return one_delete_away

    def substitutes(word):
        one_sub_away = []
        for model_word in self.language_model.vocabulary:
            _, alignment = self.channel_model.align(word, new_word)
            # If the length is too long, this word doesn't work
            if len(alignment) != len(word):
                break
            num_subs = 0
            for pair in alignment:
                if pair[0] != pair[1]:
                    if '%' in pair:
                        num_subs = -1
                        break
                    else:
                        num_subs += 1
                if num_subs == 1:
                    one_sub_away.append(model_word)
        return one_sub_away

    def generate_candidates_recurse(word_list, max_distance):
        if max_distance == 0:
            return word_list
        new_list = []
        for i.lowercase() in word_list:
            insert_words = inserts(i)
            delete_words = deletes(i)
            sub_words = subs(i)
            new_list.append(insert_words)
            new_list.append(delete_words)
            new_list.append(sub_words)
        return generate_candidates_recurse(new_list, max_distance -1)

    def generate_candidates(word):
        return generate_candidates_recurse([word.lowercase()], self.max_distance)

    def check_sentence(sentence, fallback=False):
        return_list = []
        for i in sentence:
            if i in LanguageModel:
                return_list.append([i])
            candidates = generate_candidates(i)
            if candidates == []:
                if fallback:
                    return_list.append([i])
                    break
                else:
                    return_list.append([])
                    break
            candidates = sorted(candidates, key = self.unigram_score(x) + cm_score(i, x), reverse = True)
            return_list.append(candidates)
        return return_list

    def check_text(text, fallback=False):
        doc = nlp(text)
        sentences = doc.sents
        result = []
        for sentence in sentences:
            checked_sentence = check_sentence(sentence)
            result += checked_sentence
        return result

    def autocorrect_sentence(sentence):
        possibilities = check_sentence(sentence, True)
        possibilities = [x[0] for x in possibilities]
        return possibilities

    def autocorrect_line(line):
        doc = nlp(line)
        sentences = doc.sents
        result = []
        for sentence in sentences:
            checked_sentence = autocorrect_sentence(sentence)
            result += checked_sentence
        return result

    def suggest_sentence(sentence, max_suggestions):
        possibilities = check_sentence(sentence, True)
        return_list = []
        for i in possibilities:
            if len(i) == 1:
                return_list.append(i[0])
            else:
                return_list.append(i[:max_suggestions])
        return return_list

    def suggest_text(text, max_suggestions):
        doc = nlp(text)
        sentences = doc.sents
        result = []
        for sentence in sentences:
            checked_sentence = suggest_sentence(sentence)
            result += checked_sentence
        return result
