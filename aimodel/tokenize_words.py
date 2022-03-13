#!/usr/bin/env python3
# tokenize_words.py

# Improved by: Matthew Wright, CPA, PSM, ACA
# Initial utilization allowed under the MIT license.

# third party imports
from tensorflow import data as tf_data
from keras import preprocessing as keras_preprocessing

import tensorflow_datasets as tfds

# local application and library specific imports
from aimodel.load_data import LoadJsonIntents 

class TokenizeFileData:
    """ 
    Base class 
    Tokenize all questions and answer sets within file data
    
    recommended usage steps:
    commence_tokenizer_for_~filetypechoice~
    tokenize_questions_and_answers
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.file_contents = None
        self.tokenizer = None
        self.starting_token = None
        self.ending_token = None
        self.vocabulary_size = None
        
        self.batch_size = 64
        self.buffer_size = 20000
        self.dataset = None
        
    def commence_tokenizer_for_JSON_Intents(self):
        """
        Utilize this tokenizer if you are using a JSON file
        with correlating intents to train your AI
        """
        self.file_contents = LoadJsonIntents(self.file_name)
        self.file_name = f'{self.file_name}_intents.json'
        self.file_contents.load_json_sentences_and_intents()
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder\
                         .build_from_corpus(self.file_contents.questions
                                            + self.file_contents.answers,
                                            target_vocab_size=2 ** 13)
        self.starting_token = [self.tokenizer.vocab_size]
        self.ending_token = [self.tokenizer.vocab_size + 1]
        
        # Vocabulary size plus start and end token
        self.vocabulary_size = self.tokenizer.vocab_size + 2 
        
        print(f'Intent determination finder based on file: {self.file_name}')
        
    def tokenize_questions_and_answers(self):
        """ Transform the overall dataset into a tokenized format """
        self.file_contents.questions, self.file_contents.answers = \
        self._tokenize_and_filter(self.file_contents.questions, 
                                 self.file_contents.answers)

        print('Number of words in the vocabulary set: {}'\
              .format(self.vocabulary_size))
        print('Total number of individual phrase samples: {}'\
              .format(len(self.file_contents.questions)))

        # decoder inputs use the previous target as input
        # remove starting_token from targets
        self.dataset = tf_data.Dataset.from_tensor_slices(({
                       'inputs': self.file_contents.questions,
                       'dec_inputs': self.file_contents.answers[:, :-1]
                       },
                       {'outputs': self.file_contents.answers[:, 1:]}
                       ,))

        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.shuffle(self.buffer_size)
        self.dataset = self.dataset.batch(self.batch_size)
        self.dataset = self.dataset.prefetch(tf_data.experimental.AUTOTUNE)

    def _tokenize_and_filter(self, inputs, outputs):
        """ Transform each input and output into a tokenized format """
        tokenized_questions, tokenized_answers = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = self.starting_token \
                        + self.tokenizer.encode(sentence1) \
                        + self.ending_token
            sentence2 = self.starting_token \
                        + self.tokenizer.encode(sentence2) \
                        + self.ending_token

            # check tokenized sentence length
            if len(sentence1) \
               <= self.file_contents.max_sentence_length \
               and len(sentence2) \
               <= self.file_contents.max_sentence_length:
                    tokenized_questions.append(sentence1)
                    tokenized_answers.append(sentence2)

        # pad tokenized sentences
        tokenized_questions = keras_preprocessing.sequence.pad_sequences(
            tokenized_questions, 
            maxlen=self.file_contents.max_sentence_length, padding='post')
        tokenized_answers = keras_preprocessing.sequence.pad_sequences(
            tokenized_answers, 
            maxlen=self.file_contents.max_sentence_length, padding='post')

        return tokenized_questions, tokenized_answers