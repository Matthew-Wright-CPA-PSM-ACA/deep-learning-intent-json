#!/usr/bin/env python3
# python main.py

# Created by: Matthew Wright, CPA, PSM, ACA

"""
Artificial Intelligence chatbot using Keras
Currently structured to determine intents of user input

You need to train neurons prior to running the chat

For more information, review README.md 
"""

# local application and library specific imports
from intentchat.chatbot_for_intents import ChatBot
from intenttrain.train_intents import TrainBot
from aimodel.transformer_model import KerasTransformerModel
from aimodel.tokenize_words import TokenizeFileData

if __name__ == "__main__":
    
    #basic program settings 
    chat_practice_with_bot = False # False commences training neurons
    create_new_weight_set = True # True creates new set from scratch
    file_name = "testA" #enter test file name here such as 'testA'
    
    ai_prep = TokenizeFileData(file_name) 
    ai_prep.commence_tokenizer_for_JSON_Intents()
    ai_prep.tokenize_questions_and_answers()
    
    nlp_model = KerasTransformerModel(ai_prep)
    nlp_model.create_model()
        
    if chat_practice_with_bot:
        # Terminal chat to determine intents from user input 
        nlp_model.model.load_weights(f"modelweights/{file_name}_weights.h5")
    
        intentobot = ChatBot(nlp_model)
        intentobot.chat_with_ai_for_intent()
    
    else:
        # Train initial A.I. neurons and create weights
        # set initial AI transformer model training settings here
        # Base settings: 120 epoch, 3 cycles, 2 seconds, file_name testA"""
        intentobot = TrainBot(epochs=120, cycles=3, 
                              time_delay_seconds_between_answers=5, 
                              file_name=file_name)
        
        if not create_new_weight_set:
            intentobot.reload_ai_neurons_prior_to_additional_training(nlp_model)
        
        intentobot.train_ai_neurons(ai_prep, nlp_model)
        

