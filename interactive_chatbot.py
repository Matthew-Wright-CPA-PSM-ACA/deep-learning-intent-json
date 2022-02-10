# python interactive_chatbot.py
# please ensure to run python train_intents.py prior to running this

from prep_intents.py import *
model.load_weights(f"model_weights/{fileName}_weights.h5")

class chatBot:
	exit_commands = ("quit", "exit", "stop")

	def prompt_user(self, text=None):
		if text is None:
			text = ""
		return input(text + "\n~")

	def commence_chat(self):
		inputitem = input("I'm an A.I. that's trying to figure out the intent of your question or comment.\nPlease input something:\n~ ")
		self.chat(inputitem)
		return

	def chat(self, reply):
		while not self.depart(reply):
			reply = input(predict(reply, quiet=True)+"\n~ ")
		return

	def depart(self, reply):
		for exit_command in self.exit_commands:
			if exit_command in reply:
				print("Well that was just swell. Till next time!\n\n")
				return True
		return False

intentobot = chatBot()
intentobot.commence_chat()