#!/usr/bin/env python3
# train_intents.py

from ai_model.prep_intents import *

from matplotlib import pyplot as plt
import random

from time import sleep as time_sleep

from ai_model.process_sentences import *

class trainBot:
	def __init__(self, epochs=120, cycles=3, timeDelayBetweenAnswers=9, fileName="testA"):
		
		# initial program settings
		# AI settings
		self.epochs = epochs   
		self.cycles = cycles 
		self.train_loss = []
		self.phrase = sentence_process()
		
		# question settings
		self.timeDelayBetweenAnswers = timeDelayBetweenAnswers  # seconds to delay between answer groups
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
				  "Let's have some fun this weekend!"]
		
		#file settings
		self.fileName = fileName
		
	def train_intent_neurons(self):	
		# in case of reloading the weights rather than retraining them
		#model.load_weights(f"model_weights/{self.fileName}_weights.h5")

		for x in range(self.cycles):
			print(f'Start with epoch: {x * self.epochs + 1} , having {self.cycles * self.epochs} epochs total')
			fit_model = model.fit(dataset, epochs=self.epochs)
			model.save_weights(f'model_weights/{self.fileName}_weights.h5')
			print("\n>>Testing model prediction:<<")
			for tquestion in self.test_questions:
				self.phrase.sentence = tquestion
				self.phrase.prep_intent_sentence()
				predict(self.phrase.sentence)

			time_sleep(self.timeDelayBetweenAnswers) #take a breather to review question responses

			self.train_loss.extend(fit_model.history['loss'])

			plotX = range(1, (x+1)*self.epochs+1)

			plt.figure()
			plt.plot(plotX, self.train_loss, 'blue', label="Training Loss")
			plt.legend()
			plt.savefig("cumul_err.png")