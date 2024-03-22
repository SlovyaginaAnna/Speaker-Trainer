import nltk
from nltk import word_tokenize, FreqDist


class FillerWordsAndPhrases:
    params = {
        "word_count_multiplier": 0.1,  # multiplier for most common word or phrase occurrence to be compared with others
        "occurrence_percentage": 0.0001,  # minimal percentage for word or phrase to be considered common
        "parasites": ["просто", "вот", "ну", "короче", "типа", "пожалуй", "кстати", "вообще", "буквально", "скажем",
                      "блин", "допустим", "черт", "вроде", "круто", "прикинь", "прикиньте", "реально", "отпад",
                      "отпадно", "клево", "капец", "норм", "слушай", "конечно", "наверное", "вероятно", "кажется"]
    }

    def __init__(self, cleaned_transcription):
        """
        Initialization of filler words detection class
        @param cleaned_transcription: text transcription without punctuation marks
        """
        self.cleaned_transcription = cleaned_transcription

    def count_occurrences(self, min_len=5):
        """
        Counts two-words phrases occurrences
        @param min_len: minimal length for phrase to be considered
        @return: list of two-element lists, each with phrase and its occurrence
        """
        pairs = dict()
        words = self.cleaned_transcription.split()
        for i in range(len(words) - 1):
            phrase = words[i] + ' ' + words[i + 1]
            if len(phrase) > min_len:
                if phrase not in pairs:
                    pairs[phrase] = 0
                pairs[phrase] += 1
        phrase_dic = list(pairs.items())
        phrases = sorted(phrase_dic, key=lambda x: -x[1])
        return phrases

    def find_worst_phrases(self, phrases):
        """
        Takes most common phrases from all
        @param phrases: all two-word phrases
        @return: dictionary with key - phrases and value - their occurrences
        """
        num_words = len(self.cleaned_transcription)
        max_repeats = phrases[0][1]
        if max_repeats == 1 or max_repeats / num_words < self.params["occurrence_percentage"]:
            return dict()
        # Maximal deviation from most common word or phrase occurrence
        diff = round(max_repeats * self.params["word_count_multiplier"])
        worst_word_pairs = dict()
        for word_pair, cnt in phrases:
            if cnt >= max_repeats - diff and cnt / num_words >= self.params["occurrence_percentage"]:
                worst_word_pairs[word_pair] = cnt
        return worst_word_pairs

    def get_one_words(self):
        """
        Counts all filler words from params parasites
        @return: frequency dictionary with key - words and value - their occurrences
        """
        text_tokens = word_tokenize(self.cleaned_transcription)
        text_tokens = [token.strip() for token in text_tokens if token in set(self.params["parasites"])]
        text = nltk.Text(text_tokens)
        fdist = FreqDist(text)
        return fdist

    def find_worst_words(self, fdist):
        """
        Takes most common filler words from all
        @param fdist: frequency dictionary with key - words and value - their occurrences
        @return: dictionary with key - words and value - their occurrences
        """
        num_words = len(self.cleaned_transcription)
        if len(fdist) == 0:
            return dict()
        max_repeats = fdist.most_common(1)[0][1]
        if max_repeats == 1 or max_repeats / num_words < self.params["occurrence_percentage"]:
            return dict()
        # Maximal deviation from most common word or phrase occurrence
        diff = round(max_repeats * self.params["word_count_multiplier"])
        idx = 1
        while idx <= len(fdist) and fdist.most_common(idx)[-1][1] >= max_repeats - diff and \
                fdist.most_common(idx)[-1][1] / num_words >= self.params["occurrence_percentage"]:
            idx += 1
        worst_words = dict(fdist.most_common(idx - 1))
        return worst_words

    def get_filler_words_final(self):
        """
        Concatenates all words and phrases into two dictionaries - all and most common filler words
        @return: two dictionaries with words / phrases and their occurrences
        """
        phrases = self.count_occurrences()
        worst_phrases = self.find_worst_phrases(phrases)

        fdist = self.get_one_words()
        worst_words = self.find_worst_words(fdist)

        total_dict = dict(worst_phrases) | dict(fdist)
        worst_dict = dict(worst_phrases) | dict(worst_words)
        return total_dict, worst_dict
