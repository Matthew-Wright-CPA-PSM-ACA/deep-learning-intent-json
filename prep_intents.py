# prep_intents.py

# AI Transistor model implemented for core AI - utilization allowed under the MIT license. 

from re import sub as re_sub, search as re_search
from time import sleep as time_sleep

from tensorflow import cast as tf_cast, shape as tf_shape, float32 as tf_float32, int32 as tf_int32, ones as tf_ones
from tensorflow import matmul as tf_matmul, data as tf_data, reshape as tf_reshape, transpose as tf_transpose, newaxis as tf_newaxis
from tensorflow import linalg as tf_linalg, maximum as tf_maximum, pow as tf_pow, range as tf_range, math as tf_math, nn as tf_nn
from tensorflow import concat as tf_concat, not_equal as tf_not_equal, multiply as tf_multiply, reduce_mean as tf_reduce_mean
from tensorflow import expand_dims as tf_expand_dims, argmax as tf_argmax, squeeze as tf_squeeze, random as tf_random
from tensorflow import equal as tf_equal

from keras import layers as keras_layers, Input as keras_Input
from keras import Model as keras_Model, backend as keras_backend, losses as keras_losses
from keras import optimizers as keras_optimizers, metrics as keras_metrics, preprocessing as keras_preprocessing

from tensorflow.keras.optimizers import schedules as tf_keras_optimizers_schedules
from tensorflow.keras.optimizers import Adam as tf_keras_optimizers_Adam

import tensorflow_datasets as tfds

import sys # *&*
from json import loads as json_loads, load as json_load # *&*

fileName = 'testA'   #enter test file name here  such as 'testC'
corpus = []
combo_list = []

fileType = "json"  #options are json 
jsonFileLoc = f'intent_files/{fileName}_intents.json'  

tf_random.set_seed(827)

print(f'Intent determination finder based on json file: {fileName}_intents.json')

MAX_LENGTH = 0

def preprocess_sentence(sentence, debug=False, get_size=False):
	sentence = sentence.lower().strip() 
	# removing twitter handles and links:
	tokenized_sentence = [t.strip() for t in sentence.split() if "@" not in t and "https://" not in t]
	sent_length = len(tokenized_sentence)

	sentence = " ".join(tokenized_sentence)
	# creating a space between a word and the punctuation following it
	#eg: "he is a boy." => "he is a boy ."
	sentence = re_sub(r"([?.!,])", r" \1 ", sentence)
	sentence = re_sub(r"\.\.+", " ", sentence)
	sentence = re_sub(r'[\s]+', " ", sentence)
	
	# replacing everything with space except (a-z, A-Z, "'") # *&* Simplify for intent determination only
	sentence = re_sub(r"[^a-zA-Z?']+", " ", sentence)		# *&* 
	
	# *&* remove typical 'filler' words that can overcomplicate intent determination
	removewords = ["to", "a", "at", "the", "you", "she", "he", "but", "him", "her", "it"]
	testwords = sentence.split()
	resultwords  = [word for word in testwords if word.lower() not in removewords]
	sentence = ' '.join(resultwords)
	
	sentence = sentence.strip()
	
	# adding tokenized start and end to the sentence
	if get_size:
		return sentence, sent_length
	return sentence


questions = []
answers = []

if fileType == "json":
	data_file = open(jsonFileLoc).read()
	intents = json_loads(data_file)
	
	for intent in intents['intents']:
		try:
			intentVal = str(intent['tag'])
			intentVal, size = preprocess_sentence(intentVal, get_size=True)

			if size > MAX_LENGTH:
				MAX_LENGTH = size

			#print(f'\nThe intent is: {intentVal} \n')

			for pattern in intent['patterns']:

				pattern = str(pattern)
				pattern, size = preprocess_sentence(pattern, get_size=True)

				if size > MAX_LENGTH:
					MAX_LENGTH = size

				questions.append(pattern)
				answers.append(intentVal)
				
		except Exception as err: 
			print(f'Error in trying to add question: {questions[-1]} and intent answer: {answers[-1]}')
				
	#print(f'Question is {questions} and the intent is {answers}')
					
# *** *&* end of elif fileType == "json":

# *** *&*
#print(answers)

#for intent in intents['intents']:
#	if intent['tag'] == 'acquaintance' or intent['tag'] == 'annoying':
#		print(intent['tag'])
#		print(intent['patterns'])

#if fileType == "json":
#	sys.exit() #exit the program. Simple way.
# *** *&*

orig_questions = questions
orig_answers = answers 

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2 ** 13)
# Note: tfds.features.text is depreciated.
#~ tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(questions + answers, target_vocab_size=2 ** 13)
# Using a beginning and end token to indicate a sentence beginning and ending of a sentence
BEGIN_TOKEN, ENDING_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
# Vocabulary size plus start and end token
VOCAB_SIZE = tokenizer.vocab_size + 2

def tokenize_and_filter(inputs, outputs):
	tokenized_questions, tokenized_answers = [], []

	for (sentence1, sentence2) in zip(inputs, outputs):
		# tokenize sentence
		sentence1 = BEGIN_TOKEN + tokenizer.encode(sentence1) + ENDING_TOKEN
		sentence2 = BEGIN_TOKEN + tokenizer.encode(sentence2) + ENDING_TOKEN
		
		# check tokenized sentence length
		if len(sentence1) <= MAX_LENGTH and len(
			sentence2) <= MAX_LENGTH:
			tokenized_questions.append(sentence1)
			tokenized_answers.append(sentence2)

	# pad tokenized sentences
	tokenized_questions = keras_preprocessing.sequence.pad_sequences(
		tokenized_questions, maxlen=MAX_LENGTH, padding='post')
	tokenized_answers = keras_preprocessing.sequence.pad_sequences(
		tokenized_answers, maxlen=MAX_LENGTH, padding='post')

	return tokenized_questions, tokenized_answers

questions, answers = tokenize_and_filter(questions, answers)

print('Number of words in the vocabulary set: {}'.format(VOCAB_SIZE))
print('Total number of individual phrase samples: {}'.format(len(questions)))

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# decoder inputs use the previous target as input
# remove BEGIN_TOKEN from targets
dataset = tf_data.Dataset.from_tensor_slices(({
		'inputs': questions,
		'dec_inputs': answers[:, :-1]
	},
	{'outputs': answers[:, 1:]}
	,))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf_data.experimental.AUTOTUNE)


def scaled_dot_product_attention(query, key, value, mask):
	"""Calculate the attention weights. """
	matmul_qk = tf_matmul(query, key, transpose_b=True)

	# scale matmul_qk
	depth = tf_cast(tf_shape(key)[-1], tf_float32)
	logits = matmul_qk / tf_math.sqrt(depth)

	# add the mask zero out padding tokens.
	if mask is not None:
		logits += (mask * -1e9)

	# softmax is normalized on the last axis (seq_len_k)
	attention_weights = tf_nn.softmax(logits, axis=-1)

	return tf_matmul(attention_weights, value)


class MultiHeadAttention(keras_layers.Layer):
	
	def __init__(self, d_model, num_heads, name="multi_head_attention"):
		super(MultiHeadAttention, self).__init__(name=name)
		self.num_heads = num_heads
		self.d_model = d_model

		assert d_model % self.num_heads == 0

		self.depth = d_model // self.num_heads

		self.query_dense = keras_layers.Dense(units=d_model)
		self.key_dense = keras_layers.Dense(units=d_model)
		self.value_dense = keras_layers.Dense(units=d_model)

		self.dense = keras_layers.Dense(units=d_model)

	def get_config(self):
		config = super(MultiHeadAttention, self).get_config()
		config.update({'num_heads': self.num_heads, 'd_model': self.d_model})
		return config

	def split_heads(self, inputs, batch_size):
		inputs = tf_reshape(
			inputs, shape=(batch_size, -1, self.num_heads, self.depth))
		return tf_transpose(inputs, perm=[0, 2, 1, 3])

	def call(self, inputs):
		query, key, value, mask = inputs['query'], inputs['key'], inputs[
			'value'], inputs['mask']
		batch_size = tf_shape(query)[0]

		# linear layers
		query = self.query_dense(query)
		key = self.key_dense(key)
		value = self.value_dense(value)

		# split heads
		query = self.split_heads(query, batch_size)
		key = self.split_heads(key, batch_size)
		value = self.split_heads(value, batch_size)

		# scaled dot-product attention
		scaled_attention = scaled_dot_product_attention(query, key, value, mask)
		scaled_attention = tf_transpose(scaled_attention, perm=[0, 2, 1, 3])

		# concatenation of heads
		concat_attention = tf_reshape(scaled_attention,
									  (batch_size, -1, self.d_model))

		# final linear layer
		outputs = self.dense(concat_attention)

		return outputs


def create_padding_mask(x):
	mask = tf_cast(tf_math.equal(x, 0), dtype=tf_float32)
	return mask[:, tf_newaxis, tf_newaxis, :]


def create_look_ahead_mask(x):
	seq_len = tf_shape(x)[1]
	look_ahead_mask = 1 - tf_linalg.band_part(
		tf_ones((seq_len, seq_len), dtype=tf_float32), -1, 0)
	padding_mask = create_padding_mask(x)
	return tf_maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(keras_layers.Layer):

	def __init__(self, position, d_model):
		super(PositionalEncoding, self).__init__()
		self.position = position
		self.d_model = d_model
		self.pos_encoding = self.positional_encoding(position, d_model)

	def get_config(self):
		config = super(PositionalEncoding, self).get_config()
		config.update({'position': self.position, 'd_model': self.d_model})
		return config

	def get_angles(self, position, i, d_model):
		angles = 1 / tf_pow(10000, (2 * (i // 2)) / tf_cast(d_model, tf_float32))
		return position * angles

	def positional_encoding(self, position, d_model):
		angle_rads = self.get_angles(
			position=tf_cast(tf_range(position)[:, tf_newaxis], dtype=tf_float32),
 			i=tf_cast(tf_range(d_model)[tf_newaxis, :], dtype=tf_float32),
			d_model=tf_cast(d_model, dtype=tf_float32))
		# apply sin to even index in the array
		sines = tf_math.sin(angle_rads[:, 0::2])
		# apply cos to odd index in the array
		cosines = tf_math.cos(angle_rads[:, 1::2])

		pos_encoding = tf_concat([sines, cosines], axis=-1)
		pos_encoding = pos_encoding[tf_newaxis, ...]
		return pos_encoding

	def call(self, inputs):
		return inputs + self.pos_encoding[:, :tf_shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
	inputs = keras_Input(shape=(None, d_model), name="inputs")
	padding_mask = keras_Input(shape=(1, 1, None), name="padding_mask")

	attention = MultiHeadAttention(
		d_model, num_heads, name="attention")({
			'query': inputs, 
			'key': inputs, 
			'value': inputs, 
			'mask': padding_mask
		})
	attention = keras_layers.Dropout(rate=dropout)(attention)
	attention = keras_layers.LayerNormalization(
		epsilon=1e-6)(inputs + attention)

	outputs = keras_layers.Dense(units=units, activation='relu')(attention)
	outputs = keras_layers.Dense(units=d_model)(outputs)
	outputs = keras_layers.Dropout(rate=dropout)(outputs)
	outputs = keras_layers.LayerNormalization(
		epsilon=1e-6)(attention + outputs)

	return keras_Model(
		inputs=[inputs, padding_mask], outputs=outputs, name=name)


def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
	inputs = keras_Input(shape=(None,), name="inputs")
	padding_mask = keras_Input(shape=(1, 1, None), name="padding_mask")

	embeddings = keras_layers.Embedding(vocab_size, d_model)(inputs)
	embeddings *= tf_math.sqrt(tf_cast(d_model, tf_float32))
	embeddings = PositionalEncoding(vocab_size, 
									d_model)(embeddings)

	outputs = keras_layers.Dropout(rate=dropout)(embeddings)

	for i in range(num_layers):
		outputs = encoder_layer(
			units=units, d_model=d_model, num_heads=num_heads, dropout=dropout, 
			name="encoder_layer_{}".format(i),
		)([outputs, padding_mask])

	return keras_Model(
		inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
	inputs = keras_Input(shape=(None, d_model), name="inputs")
	enc_outputs = keras_Input(
		shape=(None, d_model), name="encoder_outputs")
	look_ahead_mask = keras_Input(
		shape=(1, None, None), name="look_ahead_mask")
	padding_mask = keras_Input(shape=(1, 1, None), name='padding_mask')

	attention1 = MultiHeadAttention(
		d_model, num_heads, name="attention_1")(inputs={
		'query': inputs,
		'key': inputs,
		'value': inputs,
		'mask': look_ahead_mask
	})
	attention1 += tf_cast(inputs, dtype=tf_float32)
	attention1 = keras_layers.LayerNormalization(epsilon=1e-6)(attention1)

	attention2 = MultiHeadAttention(
		d_model, num_heads, name="attention_2")(inputs={
		'query': attention1,
		'key': enc_outputs,
		'value': enc_outputs,
		'mask': padding_mask
	})
	attention2 = keras_layers.Dropout(rate=dropout)(attention2)
	attention2 = keras_layers.LayerNormalization(
		epsilon=1e-6)(attention2 + attention1)

	outputs = keras_layers.Dense(units=units, activation='relu')(attention2)
	outputs = keras_layers.Dense(units=d_model)(outputs)
	outputs = keras_layers.Dropout(rate=dropout)(outputs)
	outputs = keras_layers.LayerNormalization(
		epsilon=1e-6)(outputs + attention2)

	return keras_Model(
		inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
		outputs=outputs,
		name=name)


def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
	inputs = keras_Input(shape=(None,), name='inputs')
	enc_outputs = keras_Input(
		shape=(None, d_model), name='encoder_outputs')
	look_ahead_mask = keras_Input(
		shape=(1, None, None), name='look_ahead_mask')
	padding_mask = keras_Input(shape=(1, 1, None), name='padding_mask')

	embeddings = keras_layers.Embedding(vocab_size, d_model)(inputs)
	embeddings *= tf_math.sqrt(tf_cast(d_model, tf_float32))
	embeddings = PositionalEncoding(vocab_size, 
									d_model)(embeddings)

	outputs = keras_layers.Dropout(rate=dropout)(embeddings)

	for i in range(num_layers):
		outputs = decoder_layer(units=units, d_model=d_model, num_heads=num_heads, dropout=dropout,
			name='decoder_layer_{}'.format(i),
		)(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

	return keras_Model(
		inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
		outputs=outputs,
		name=name)


def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
	inputs = keras_Input(shape=(None,), name="inputs")
	dec_inputs = keras_Input(shape=(None,), name="dec_inputs")

	enc_padding_mask = keras_layers.Lambda(
		create_padding_mask, output_shape=(1, 1, None),
		name='enc_padding_mask')(inputs)
	# mask the future tokens for decoder inputs at the 1st attention block
	look_ahead_mask = keras_layers.Lambda(
		create_look_ahead_mask,
		output_shape=(1, None, None),
		name='look_ahead_mask')(dec_inputs)
	# mask the encoder outputs for the 2nd attention block
	dec_padding_mask = keras_layers.Lambda(
		create_padding_mask, output_shape=(1, 1, None),
		name='dec_padding_mask')(inputs)

	enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers,	units=units, d_model=d_model,
		num_heads=num_heads,dropout=dropout,)(inputs=[inputs, enc_padding_mask])

	dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers,	units=units,d_model=d_model,
		num_heads=num_heads,dropout=dropout)(
		inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

	outputs = keras_layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

	return keras_Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

def predict(sentence, quiet=False):
	prediction = evaluate(sentence)
	
	predicted_sentence = tokenizer.decode(
		[i for i in prediction if i < tokenizer.vocab_size])
	
	# *&* roughly equivalent to
	#predicted_sentence = []
		#for x in prediction:
    		#if i < tokenizer.vocab_size:
        		#predicted_sentence.append(i)
	
	if not quiet:
		print('User Input: {}'.format(sentence))
		print('Matching intent: {}'.format(predicted_sentence))

	return predicted_sentence

def evaluate(sentence):
	sentence = preprocess_sentence(sentence)

	input_sent = BEGIN_TOKEN + tokenizer.encode(sentence) + ENDING_TOKEN

	sentence = tf_expand_dims(input_sent, axis=0)

	output = tf_expand_dims(BEGIN_TOKEN, 0)
	
	for i in range(MAX_LENGTH):
		predictions = model(inputs=[sentence, output], training=False)
		
		predictions = predictions[:, -1:, :]
		predicted_id = tf_cast(tf_argmax(predictions, axis=-1), tf_int32)
				
		if tf_equal(predicted_id, ENDING_TOKEN[0]):
			break
			
		output = tf_concat([output, predicted_id], axis=-1)

	return tf_squeeze(output, axis=0)


keras_backend.clear_session()

NUM_LAYERS = 2 
D_MODEL = 256   
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1   

model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, units=UNITS, 
							 d_model=D_MODEL, num_heads=NUM_HEADS,	dropout=DROPOUT)


def loss_function(y_true, y_pred):
	y_true = tf_reshape(y_true, shape=(-1, MAX_LENGTH - 1))
	loss = keras_losses.SparseCategoricalCrossentropy(
		from_logits=True, reduction='none')(y_true, y_pred)

	mask = tf_cast(tf_not_equal(y_true, 0), tf_float32)
	loss = tf_multiply(loss, mask)

	return tf_reduce_mean(loss)


class CustomSchedule(tf_keras_optimizers_schedules.LearningRateSchedule):

	def __init__(self, d_model, warmup_steps=4000):
		super(CustomSchedule, self).__init__()

		self.d_model = d_model
		self.d_model = tf_cast(self.d_model, tf_float32)

		self.warmup_steps = warmup_steps

	def get_config(self):
		return {
			'd_model': self.d_model,
			'warmup_steps': self.warmup_steps
		}
		return config

	def __call__(self, step):
		arg1 = tf_math.rsqrt(step)
		arg2 = step * (self.warmup_steps ** -1.5)

		return tf_math.rsqrt(self.d_model) * tf_math.minimum(arg1, arg2)


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf_keras_optimizers_Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


def accuracy(y_true, y_pred):
	y_true = tf_reshape(y_true, shape=(-1, MAX_LENGTH - 1))
	return keras_metrics.sparse_categorical_accuracy(y_true, y_pred)


model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

