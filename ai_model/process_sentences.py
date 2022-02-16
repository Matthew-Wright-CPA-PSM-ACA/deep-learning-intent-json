#!/usr/bin/env python3
# process_sentences.py
# Matthew Wright, CPA, PSM, ACA

from re import sub as re_sub, search as re_search

class sentence_process:
	def __init__(self):
		self.sentence = ""
		self.tokenized_sentence = ""
		self.sent_length = 0
		
	#def prep_intent_sentence(self, sentence, debug=False, get_size=False):
	def prep_intent_sentence(self):
		self.sentence = self.sentence.lower().strip() 
		# removing twitter handles and links:
		self.tokenized_sentence = [t.strip() for t in self.sentence.split() if "@" not in t and "https://" not in t]
		self.sent_length = len(self.tokenized_sentence)

		self.sentence = " ".join(self.tokenized_sentence)
		# creating a space between a word and the punctuation following it
		#eg: "he is a boy." => "he is a boy ."
		self.sentence = re_sub(r"([?.!,])", r" \1 ", self.sentence)
		self.sentence = re_sub(r"\.\.+", " ", self.sentence)
		self.sentence = re_sub(r'[\s]+', " ", self.sentence)

		# replacing everything with space except (a-z, A-Z, "'") # *&* Simplify for intent determination only
		self.sentence = re_sub(r"[^a-zA-Z?']+", " ", self.sentence)		# *&* 

		# *&* remove typical 'filler' words that can overcomplicate intent determination
		removewords = ["to", "a", "at", "an", "the", "you", "she", "he", "we", "but", "him", "her", "it", "i", "of"]
		testwords = self.sentence.split()
		resultwords  = [word for word in testwords if word.lower() not in removewords]
		self.sentence = ' '.join(resultwords)

		self.sentence = self.sentence.strip()
