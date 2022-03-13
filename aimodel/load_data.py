#!/usr/bin/env python3
# load_data.py

# Created by: Matthew Wright, CPA, PSM, ACA

# standard library imports
from sys import exit as sys_exit
from json import loads as json_loads, load as json_load

# local application and library specific imports
from aimodel.process_sentences import IntentSentencePreProcess
from mylibrary.context_managers import JSONContextManager

class LoadLanguageExamples:
    """ 
    Base class 
    Load the data files in different format into the question and answer lists
    """
    def __init__(self, file_name):
        self.questions = []
        self.answers = []
        self.file_name = file_name
        self.max_sentence_length = 0

class LoadJsonIntents(LoadLanguageExamples):
    """ 
    Derived class: is a LoadLanguageExamples 
    Loads the JSON data files into the question and answer lists
    question list contains the entire sentence
    answer list contains the correlating intent of sentence
    
    file_name = Need to pass the start of JSON file name, such as testA
    """
    def __init__(self, file_name):
        """ Need to pass file_name through to base class to commence """
        super().__init__(file_name)
        self.jsonFileLoc = f'intentfiles/{file_name}_intents.json'
        self.sentence_matching_intent = None
        self.process_phrase = IntentSentencePreProcess()
        
    def load_json_sentences_and_intents(self):
        """
        load the sentences and corresponding intents that are stored 
        within the json file for use in training the A.I. 
        """
        with JSONContextManager(self.jsonFileLoc, 'r') as data_file:
            self.sentence_matching_intent = json_load(data_file)
        
        self._prepare_then_separate_sentences_and_intents()
        
    def _prepare_then_separate_sentences_and_intents(self):
        """
        load the sentences and corresponding intents that are stored 
        within the json file for use in training the A.I. 
        """
        for intent in self.sentence_matching_intent['intents']:
            try: 
                self.process_phrase.sentence = str(intent['tag'])
                self.process_phrase.prepare_sentence_for_tokenization()
                intent_value = self.process_phrase.sentence

                if self.process_phrase.sentence_length > self.max_sentence_length:
                    self.max_sentence_length = self.process_phrase.sentence_length

                for pattern in intent['patterns']:
                    self.process_phrase.sentence = str(pattern)
                    self.process_phrase.prepare_sentence_for_tokenization()

                    if self.process_phrase.sentence_length > self.max_sentence_length:
                        self.max_sentence_length = self.process_phrase.sentence_length

                    self.questions.append(self.process_phrase.sentence)
                    self.answers.append(intent_value)

            except Exception as err: 
                error_string = f'Error in trying to add question: {self.questions[-1]}'
                error_string += f'and intent answer: {self.answers[-1]}'
                print(error_string)
                #print(f'Error in trying to add question: {self.questions[-1]} and intent answer: {self.answers[-1]}')

    def list_all_intents_in_json(self):
        """
        This will traverse all of the json sentence and intent parts
        and will list only the current intents
        after it will exit the program
        """
        temp_intents = []
        
        for x in self.answers:
            if x not in temp_intents:
                temp_intents.append(x)   
        
        print(f'The current intents are: {temp_intents}')
    
        sys_exit() #exit the program. Simple way
        
        
