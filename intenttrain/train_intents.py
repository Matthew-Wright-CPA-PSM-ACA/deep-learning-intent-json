#!/usr/bin/env python3
# train_intents.py

# Matthew Wright, CPA, PSM, ACA

# standard library imports
from time import sleep as time_sleep

# third party imports
from matplotlib import pyplot as plt

# local application and library specific imports
from aimodel.process_sentences import IntentSentencePreProcess


class TrainBot:
    """ 
    Utilize this class to train the AI 
    through its epochs times the # of cycles 
    """
    def __init__(self, 
                 epochs=120, 
                 cycles=3, 
                 time_delay_seconds_between_answers=9, 
                 file_name="testA"):
        
        #initial AI settings
        self.epochs = epochs   
        self.cycles = cycles 
        self.train_loss = []
        self.phrase = IntentSentencePreProcess()
        
        # question settings
        self.time_delay_seconds_between_answers = time_delay_seconds_between_answers 
        self.test_questions = ["Let's go out downtown and have fun",
                              "what do feel like doing this weekend?",
                              "I like the color blue",  
                              "hello what are you doing",
                              "Let's go for a coffee", 
                              "Do you watch football?",
                              "What do you do for fun?",
                              "Goodnight, i'm so tired i'm going to bed",
                              "What is your favorite color?",
                              "Is it your birthday today?",
                              "Let's have some fun this weekend!"
                              ]
        
        #file settings
        self.file_name = file_name
        self.model_fit = None
        
    def reload_ai_neurons_prior_to_additional_training(self, ai_model): 
        """
        Utilize this to first reload the model weights
        and then continue with additional A.I. training
        rather than retraining them from the start each time
        """
        ai_model.model.load_weights(f"modelweights/{self.file_name}_weights.h5")
        

    def train_ai_neurons(self, ai_prep, ai_model):
        """ 
        train the A.I. deep learning neurons by cycling through 
        the epochs, times the number cycles to complete the training
        """
        for cycles_range in range(self.cycles):
            print(f"Starting with epoch: {cycles_range * self.epochs + 1}")
            print(f" Total epoch's are : {self.cycles * self.epochs}")
            self.model_fit = ai_model.model.fit(ai_prep.dataset, epochs=self.epochs)
            ai_model.model.save_weights(f'modelweights/{self.file_name}_weights.h5')
            
            self._test_model_prediction_using_test_questions(ai_model)
            self.graph_plot_epochs_and_training_loss(cycles_range)
            
    def _test_model_prediction_using_test_questions(self, ai_model):
        """
        test the actual AI neuron model predictions
        by cycling through test_questions
        and output responses to terminal screen
        """
        print("\n----- A.I. prediction of matching testing model -----")
        for tquestion in self.test_questions:
            self.phrase.sentence = tquestion
            self.phrase.prepare_sentence_for_tokenization()
            ai_model.predict_matching_sentence_response(self.phrase.sentence)
        
        #allow X seconds for prediction answer review
        time_sleep(self.time_delay_seconds_between_answers) 
        
    def graph_plot_epochs_and_training_loss(self, cycles_range):
        """ 
        use matplotlib to plot epoch and training loss of AI training 
        This will then save a .png plot to same directory as main.py
        """
        self.train_loss.extend(self.model_fit.history['loss'])

        plotX = range(1, (cycles_range+1)*self.epochs+1)

        plt.figure()
        plt.plot(plotX, self.train_loss, 'blue', label="Training Loss")
        plt.legend()
        plt.savefig("cumul_err.png")
        
        