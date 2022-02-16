#!/usr/bin/env python3
# python main.py
"""
Main script. See README.md 
"""

from ai_model.prep_intents import *
from intent_chat.chatbot_for_intents import *
from intent_train.train_intents import *

if __name__ == "__main__":
	
	#basic settings
	trainorchat = 0 #0 is train, 1 is to chat
	fileName = "testA"
	
	#intent chatting
	if trainorchat == 1:
		
		model.load_weights(f"model_weights/{fileName}_weights.h5")
	
		intentobot = chatBot()
		intentobot.commence_chat()
	
	#intent training
	else:
		#set initial AI transformer model training settings here
		#intentobot = trainBot() #automatically 120 epoch, 3 cycles, 2 seconds delay, fileName testA
		intentobot = trainBot(epochs=120, cycles=3, timeDelayBetweenAnswers=7, fileName=fileName)
		intentobot.train_intent_neurons()