#!/usr/bin/env python3
# process_sentences.py

# Created by: Matthew Wright, CPA, PSM, ACA

# standard library imports
from re import sub as re_sub, search as re_search

class SentencePreProcess:
    """ 
    Base class
    Takes either a stored or user input sentence, strips the unecessary
    characters from it, alters it to a tokenize acceptable format 
    so that can proceed through a tokenized transformation 
    usable with A.I. keras deep learning neuron creation 
    """
    
    def __init__(self):
        self.sentence = None
        self.sentence_length = 0

    def __str__(self):
        """ prints out the sentence being processed """
        return 'Processing sentence {self.sentence} for tokenization'
        
    def _remove_handles_and_links(self):
        """ remove @ handles and https:// links in sentence """
        self.sentence = self.sentence.lower().strip() 
        tokenizable_sentence = [t.strip() for t in self.sentence.split() 
                                if "@" not in t and "https://" not in t]
        self.sentence_length = len(tokenizable_sentence)

        self.sentence = " ".join(tokenizable_sentence)
    
    def _create_space_between_words_and_punctuation(self):
        """
        creating a space between a word and the punctuation following it
        eg: "he is a boy." => "he is a boy ."
        """
        self.sentence = re_sub(r"([?.!,])", r" \1 ", self.sentence)
        self.sentence = re_sub(r"\.\.+", " ", self.sentence)
        self.sentence = re_sub(r'[\s]+', " ", self.sentence)


class BasicSentencePreProcess(SentencePreProcess):
    """ 
    Derived class: is a SentencePreProcess
    Class for processing sentences utilized in regular sentences matching
    for tokenization preparation in A.I. deep learning neuron matching
    """
    
    def __init__(self):
        super().__init__()
    
    def prepare_sentence_for_tokenization(self):
        """ 
        Main operating function for the class 
        prepares the actual sentence for tokenization 
        """
        try:
            self._remove_handles_and_links()
            self._create_space_between_words_and_punctuation()
        except:
            # Exception will still allow self.sentence to be populated
            # with tokenizable value - continue on 
            pass
    
class IntentSentencePreProcess(SentencePreProcess):
    """ 
    Derived class: is a SentencePreProcess
    Class for processing sentences utilized in intent determination 
    Focuses on simplifying the sentence for intent determination 
    """
    
    def __init__(self):
        super().__init__()
        self.filler_words = ["to", "a", "at", "the", 
                             "you", "she", "he", "but",
                             "him", "her", "it"
                            ]
        
    def prepare_sentence_for_tokenization(self):
        """ 
        Main operating function for the class 
        prepares the actual sentence for tokenization 
        """
        try:
            self._remove_handles_and_links()
            self._create_space_between_words_and_punctuation()
            self._remove_filler_words()
            self._replace_with_space_except_basic_characters()
            
        except:
            # Exception will still allow self.sentence to be populated
            # with tokenizable value - continue on
            print(f'exception in prepare_sentence_for_tokenization\n')
            print(f'sentence of issue: {self.sentence}')
            pass
    
    def _remove_filler_words(self):
        """remove 'filler' words that overcomplicates intent determination"""
        testwords = self.sentence.split()
        resultwords  = [word for word in testwords if word.lower()
                        not in self.filler_words]
        self.sentence = ' '.join(resultwords)

        self.sentence = self.sentence.strip()
        
    def _replace_with_space_except_basic_characters(self):
        """ 
        replacing all characters with space except (a-z, A-Z, "'") 
        for intent determination purposes
        """
        self.sentence = re_sub(r"[^a-zA-Z?']+", " ", self.sentence)
    
    