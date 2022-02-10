# python train_intents.py

from prep_intents import *

from matplotlib import pyplot as plt
import random

from time import sleep as time_sleep

# initial program settings
timeDelayBetweenAnswers = 7   # seconds to delay between answer groups

# Have the option of reloading the weights rather than retraining them
if False:
	model.load_weights(f"model_weights/{fileName}_weights.h5")

test_questions = ["Let's go out downtown and have fun",
				  "What's your favourite restaurant?",  
				  "What are you up to this weekend?",
				  "I like the color blue",  
				  "hello what are you doing",
				  "Do you watch football?",
				  "What do you do for fun?",
				  "Goodnight, i'm so tired i'm going to bed",
				  "What is your favorite color?",
				  "Is it your birthday today?",
				  "Let's have some fun this weekend!"]

epochs = 80   #started at 50    (300 epoch and 10 cycles works well --> quicker, more accurate for longer sentences)
samples = len(orig_questions)
cycles = 7  #started at 12, 30 works better
train_loss = []

for x in range(cycles):
	print(f"Start with epoch: {x * epochs + 1} , having {cycles * epochs} epochs total")
	fit_model = model.fit(dataset, epochs=epochs)
	model.save_weights(f"model_weights/{fileName}_weights.h5")
	print("\n>>Testing model prediction:<<")
	for tquestion in test_questions:
		tquestion = preprocess_sentence(tquestion, get_size=False)
		predict(tquestion)

	time_sleep(timeDelayBetweenAnswers) #take a breather to review question responses
	
	train_loss.extend(fit_model.history['loss'])
	
	plotX = range(1, (x+1)*epochs+1)

	plt.figure()
	plt.plot(plotX, train_loss, 'blue', label="Training Loss")
	plt.legend()
	plt.savefig("cumul_err.png")
