import numpy as np
import re


class NaiveBayesMessageClassifier:
    def __init__(self):
        self.neutral_probability = None
        self.negative_probability = None
        self.positive_probability = None
        self.neutral_words_dict = {}
        self.negative_words_dict = {}
        self.positive_words_dict = {}

    def fit(self, features, classes):
        self.neutral_probability = self._get_class_probabilities(classes, "neutral")
        self.positive_probability = self._get_class_probabilities(classes, "positive")
        self.negative_probability = self._get_class_probabilities(classes, "negative")

        words_tuple = self._get_words_by_classes(features, classes)
        all_words, neutral_words, positive_words, negative_words = words_tuple

        for word in np.unique(all_words):
            self.neutral_words_dict[word] = self._get_word_class_probability(
                word, self.neutral_probability, neutral_words, all_words
            )
            self.positive_words_dict[word] = self._get_word_class_probability(
                word, self.positive_probability, positive_words, all_words
            )
            self.negative_words_dict[word] = self._get_word_class_probability(
                word, self.negative_probability, negative_words, all_words
            )

    def predict(self, messages):
        prediction = []
        for message in messages:
            words = self.clean_message(message).split(" ")
            neutral_probability = self._get_probability(
                words, self.neutral_words_dict, self.neutral_probability
            )
            positive_probability = self._get_probability(
                words, self.positive_words_dict, self.positive_probability
            )
            negative_probability = self._get_probability(
                words, self.negative_words_dict, self.negative_probability
            )

            predicted_class = max(
                [
                    (neutral_probability, "neutral"),
                    (positive_probability, "positive"),
                    (negative_probability, "negative"),
                ],
                key=lambda x: x[0],
            )

            prediction.append(predicted_class[1])

        return prediction

    def _get_words_by_classes(self, messages, classes):
        neutral_words = []
        negative_words = []
        positive_words = []
        all_words = []

        for index, message in enumerate(messages):
            words = self.clean_message(message).split(" ")
            if len(words) == 0:
                continue

            if classes[index] == "neutral":
                neutral_words.extend(words)

            elif classes[index] == "negative":
                negative_words.extend(words)

            elif classes[index] == "positive":
                positive_words.extend(words)

            else:
                continue

            all_words.extend(words)

        return (
            np.array(all_words),
            np.array(neutral_words),
            np.array(positive_words),
            np.array(negative_words),
        )

    def _get_word_class_probability(
        self, word, class_probability, class_words_arr, all_words_arr
    ):
        class_count = np.count_nonzero(class_words_arr == word)
        overall_count = np.count_nonzero(all_words_arr == word)

        return self._normalize_probability(
            class_count / overall_count, class_probability, overall_count
        )

    @staticmethod
    def _get_probability(words, dictionary, class_probability):
        probability = 1
        for word in words:
            probability *= dictionary.get(word, 1)

        return probability * class_probability

    @staticmethod
    def clean_message(message):
        return re.sub("[^A-Za-z0-9`]+", " ", message).lower().strip()

    @staticmethod
    def _get_class_probabilities(classes, class_name):
        return np.count_nonzero(classes == class_name) / len(classes)

    @staticmethod
    def _normalize_probability(word_probability, class_probability, word_num):
        return (word_num * word_probability + class_probability) / (word_num + 1)
