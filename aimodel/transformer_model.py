#!/usr/bin/env python3
# prep_intents.py

# AI Transistor model implemented for core AI 
# Initial utilization allowed under the MIT license. 
# Improved by: Matthew Wright, CPA, PSM, ACA

# standard library imports
from time import sleep as time_sleep

# third party imports
from tensorflow import cast as tf_cast, shape as tf_shape, float32 as tf_float32, int32 as tf_int32, ones as tf_ones
from tensorflow import matmul as tf_matmul, reshape as tf_reshape, transpose as tf_transpose, newaxis as tf_newaxis
from tensorflow import linalg as tf_linalg, maximum as tf_maximum, pow as tf_pow, range as tf_range, math as tf_math, nn as tf_nn
from tensorflow import concat as tf_concat, not_equal as tf_not_equal, multiply as tf_multiply, reduce_mean as tf_reduce_mean
from tensorflow import expand_dims as tf_expand_dims, argmax as tf_argmax, squeeze as tf_squeeze, random as tf_random
from tensorflow import equal as tf_equal

from keras import layers as keras_layers, Input as keras_Input
from keras import Model as keras_Model, backend as keras_backend, losses as keras_losses
from keras import optimizers as keras_optimizers, metrics as keras_metrics

from tensorflow.keras.optimizers import schedules as tf_keras_optimizers_schedules
from tensorflow.keras.optimizers import Adam as tf_keras_optimizers_Adam

tf_random.set_seed(827)


class KerasTransformerModel:
    """ 
    Utilize Tensorflow 2 Keras to create a Keras deep learning
    bi-directional transformer based model 
    """
    def __init__(self, ai_prep):
        self.num_layers = 2
        self.d_model = 256
        self.num_heads = 8
        self.units = 512
        self.dropout = 0.1
        
        self.model = None
        self.learning_rate = None
        self.optimizer = None
        
        self.ai_prep = ai_prep
    
    def create_model(self):
        """ Callable function to create the Keras transformer model """
        keras_backend.clear_session()

        self.model = self._transformer_model_creation()
        
        self.learning_rate = CustomSchedule(self.d_model)

        self.optimizer = tf_keras_optimizers_Adam(self.learning_rate, 
                                                  beta_1=0.9, 
                                                  beta_2=0.98, 
                                                  epsilon=1e-9)
        
        self.model.compile(optimizer=self.optimizer, 
                           loss=self._network_loss_function, 
                           metrics=[self._accuracy])
    
    def _transformer_model_creation(self):
        """ Transformer bi-directional model core creation """
        inputs = keras_Input(shape=(None,), name="inputs")
        dec_inputs = keras_Input(shape=(None,), name="dec_inputs")

        enc_padding_mask = keras_layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='enc_padding_mask')(inputs)
        
        # mask the future tokens for decoder inputs at the 1st attention block
        look_ahead_mask = keras_layers.Lambda(
            self.create_look_ahead_mask,
            output_shape=(1, None, None),
            name='look_ahead_mask')(dec_inputs)
        
        # mask the encoder outputs for the 2nd attention block
        dec_padding_mask = keras_layers.Lambda(
            self.create_padding_mask, output_shape=(1, 1, None),
            name='dec_padding_mask')(inputs)

        enc_outputs = self._encoder()(inputs=[inputs, enc_padding_mask])

        dec_outputs = self._decoder()(inputs=[dec_inputs, 
                                              enc_outputs, 
                                              look_ahead_mask, 
                                              dec_padding_mask
                                             ])

        outputs = keras_layers.Dense(units=self.ai_prep.vocabulary_size, 
                                                name="outputs")(dec_outputs)

        return keras_Model(inputs=[inputs, dec_inputs], 
                           outputs=outputs, name="transformer")
    
    def _encoder(self):
        """
        The encoder will map a sequence of input into an abstract 
        numerical based representation which contains the entirety
        of the learned information for all the sentences input.
        """
        
        inputs = keras_Input(shape=(None,), name="inputs")
        padding_mask = keras_Input(shape=(1, 1, None), name="padding_mask")

        embeddings = keras_layers.Embedding(self.ai_prep.vocabulary_size,
                                            self.d_model)(inputs)
        embeddings *= tf_math.sqrt(tf_cast(self.d_model, tf_float32))
        embeddings = PositionalEncoding(self.ai_prep.vocabulary_size, 
                                        self.d_model)(embeddings)

        outputs = keras_layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self._encoder_layer(name="encoder_layer_{}".format(i),) \
                                    ([outputs, padding_mask])

        return keras_Model(
            inputs=[inputs, padding_mask], outputs=outputs, name="encoder")

    def _encoder_layer(self, name='encoder_layer'):
        """
        The encoding layer consists of the following sublayers:
        Multi-headed attention
        Two (2) dense layers
        A dropout occurs after each dense layers
        """
        inputs = keras_Input(shape=(None, self.d_model), name="inputs")
        padding_mask = keras_Input(shape=(1, 1, None), name="padding_mask")

        attention = MultiHeadAttention(self.d_model, 
                                       self.num_heads, 
                                       name="attention") \
                                      ({'query': inputs, 
                                        'key': inputs, 
                                        'value': inputs, 
                                        'mask': padding_mask
                                      })
        attention = keras_layers.Dropout(rate=self.dropout)(attention)
        attention = keras_layers.LayerNormalization(
            epsilon=1e-6)(inputs + attention)

        outputs = keras_layers.Dense(units=self.units, 
                                     activation='relu')(attention)
        outputs = keras_layers.Dense(units=self.d_model)(outputs)
        outputs = keras_layers.Dropout(rate=self.dropout)(outputs)
        outputs = keras_layers.LayerNormalization(
            epsilon=1e-6)(attention + outputs)

        return keras_Model(inputs=[inputs, padding_mask], 
                           outputs=outputs, 
                           name=name)

    def _decoder(self):
        """
        The encoder output is the input for the decoder
        
        The decoder takes the numerical based representation
        and generates an output sequence 
        """
        inputs = keras_Input(shape=(None,), name='inputs')
        enc_outputs = keras_Input(
            shape=(None, self.d_model), name='encoder_outputs')
        look_ahead_mask = keras_Input(
            shape=(1, None, None), name='look_ahead_mask')
        padding_mask = keras_Input(shape=(1, 1, None), name='padding_mask')

        embeddings = keras_layers.Embedding(self.ai_prep.vocabulary_size,
                                            self.d_model)(inputs)
        embeddings *= tf_math.sqrt(tf_cast(self.d_model, tf_float32))
        embeddings = PositionalEncoding(self.ai_prep.vocabulary_size, 
                                        self.d_model)(embeddings)

        outputs = keras_layers.Dropout(rate=self.dropout)(embeddings)

        for i in range(self.num_layers):
            outputs = self._decoder_layer(name='decoder_layer_{}'.format(i),)\
                                         (inputs=[outputs, 
                                                  enc_outputs, 
                                                  look_ahead_mask, 
                                                  padding_mask
                                                 ])

        return keras_Model(inputs=[inputs, 
                                   enc_outputs, 
                                   look_ahead_mask, 
                                   padding_mask],
                           outputs=outputs,
                           name='decoder')

    def _decoder_layer(self, name='decoder_layer'):
        """
        The decoding layer consists of the following sublayers:
        Masked multi-head attention, including a padding and look ahead mask
        Multi-headed attention, including a padding mask
        Two (2) dense layers
        A dropout occurs after each dense layers
        """
        inputs = keras_Input(shape=(None, self.d_model), name="inputs")
        enc_outputs = keras_Input(
            shape=(None, self.d_model), name="encoder_outputs")
        look_ahead_mask = keras_Input(
            shape=(1, None, None), name="look_ahead_mask")
        padding_mask = keras_Input(shape=(1, 1, None), name='padding_mask')

        attention1 = MultiHeadAttention(self.d_model, 
                                        self.num_heads, 
                                        name="attention_1") \
                                       (inputs={'query': inputs,
                                                'key': inputs,
                                                'value': inputs,
                                                'mask': look_ahead_mask
                                               })
        attention1 += tf_cast(inputs, dtype=tf_float32)
        attention1 = keras_layers.LayerNormalization(epsilon=1e-6)(attention1)

        attention2 = MultiHeadAttention(self.d_model, 
                                        self.num_heads, 
                                        name="attention_2") \
                                       (inputs={'query': attention1,
                                                'key': enc_outputs,
                                                'value': enc_outputs,
                                                'mask': padding_mask
                                               })
        attention2 = keras_layers.Dropout(rate=self.dropout)(attention2)
        attention2 = keras_layers.LayerNormalization(epsilon=1e-6) \
                                                    (attention2 + attention1)

        outputs = keras_layers.Dense(units=self.units, 
                                     activation='relu')(attention2)
        outputs = keras_layers.Dense(units=self.d_model)(outputs)
        outputs = keras_layers.Dropout(rate=self.dropout)(outputs)
        outputs = keras_layers.LayerNormalization(epsilon=1e-6) \
                                                 (outputs + attention2)

        return keras_Model(inputs=[inputs, 
                                   enc_outputs, 
                                   look_ahead_mask, 
                                   padding_mask
                                  ],
                           outputs=outputs,
                           name=name)

    def _network_loss_function(self, y_true, y_pred):
        """Determines the loss function during neural network training"""
        y_true = tf_reshape(y_true, 
                            shape=(-1, 
                                   self.ai_prep.file_contents. \
                                   max_sentence_length - 1))
        
        loss = keras_losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)

        mask = tf_cast(tf_not_equal(y_true, 0), tf_float32)
        loss = tf_multiply(loss, mask)

        return tf_reduce_mean(loss)

    def _accuracy(self, y_true, y_pred):
        """ Performs the accuracy calculation for the neural network """
        y_true = tf_reshape(y_true, 
                            shape=(-1, self.ai_prep.\
                                   file_contents.max_sentence_length - 1))
        return keras_metrics.sparse_categorical_accuracy(y_true, y_pred)

    def predict_matching_sentence_response(self, sentence, quiet=False):
        """
        Callable function that results in taking the input original 
        phrase in, returning the predicted sentence from the A.I. 
        NLP neural network and decoding it back to letters from
        the tokenized version
        """
        self.ai_prep.file_contents.process_phrase.sentence = sentence
        prediction = self._evaluate_tokenized_sentence()

        predicted_sentence = self.ai_prep.tokenizer.decode(
            [i for i in prediction if i < self.ai_prep.tokenizer.vocab_size])

        if not quiet:
            print('User Input: {}'.format(sentence))
            print('Matching intent: {}'.format(predicted_sentence))

        return predicted_sentence

    def _evaluate_tokenized_sentence(self):
        """
        Evaluates the sentence by preparing the sentence for 
        tokenization, encodes the sentence by tokenizing it, and runs
        the A.I. prediction model
        """
        self.ai_prep.file_contents.process_phrase. \
        prepare_sentence_for_tokenization()

        input_sent = self.ai_prep.starting_token \
                    + self.ai_prep.tokenizer.encode(self.ai_prep.file_contents \
                                                    .process_phrase.sentence) \
                    + self.ai_prep.ending_token
        
        self.ai_prep.file_contents.process_phrase.sentence = \
        tf_expand_dims(input_sent, axis=0)

        output = tf_expand_dims(self.ai_prep.starting_token, 0)

        for i in range(self.ai_prep.file_contents.max_sentence_length):
            predictions = self.model(inputs=[self.ai_prep.file_contents. \
                                             process_phrase.sentence, 
                                             output], 
                                     training=False)

            predictions = predictions[:, -1:, :]
            predicted_id = tf_cast(tf_argmax(predictions, axis=-1), 
                                   tf_int32)

            if tf_equal(predicted_id, self.ai_prep.ending_token[0]):
                break

            output = tf_concat([output, predicted_id], axis=-1)

        return tf_squeeze(output, axis=0)
    
    def create_padding_mask(self, x):
        """
        There are values to the padding sequences
        This function masks the padding values so the Transformer model
        pays no attention to them 
        """
        mask = tf_cast(tf_math.equal(x, 0), dtype=tf_float32)
        
        return mask[:, tf_newaxis, tf_newaxis, :]

    def create_look_ahead_mask(self, x):
        """
        Look ahead mask is a matrix, with identical size to
        the attention scores, filled with either -infinity or 0's
        
        This masks out future tokens, which then stops the decoder 
        from analyzing these future tokens
        """
        seq_len = tf_shape(x)[1]
        look_ahead_mask = 1 - tf_linalg.band_part(
            tf_ones((seq_len, seq_len), dtype=tf_float32), -1, 0)
        padding_mask = self.create_padding_mask(x)
        
        return tf_maximum(look_ahead_mask, padding_mask)
    

class CustomSchedule(tf_keras_optimizers_schedules.LearningRateSchedule):
    """ Optimization scheduler of the learning rates """
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf_cast(self.d_model, tf_float32)

        self.warmup_steps = warmup_steps

    def get_config(self):
        """ get the configuartion items of Custom Schedule """
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
        return config

    def __call__(self, step):
        arg1 = tf_math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf_math.rsqrt(self.d_model) * tf_math.minimum(arg1, arg2)


class MultiHeadAttention(keras_layers.Layer):
    """ 
    Equivalent to a tranditional self-attention layer, except
    multi-headed attention improves performance of the attention layer
    Transformer uses 8 attention heads
    Weight matrices are: Query / Key / Value = Q/K/V matrix
    The system keeps segregated weight matrices for each of the 8 heads
    
    Logistics
    1) take input sentence
    2) Encode each word
    3) Split it via the number of attention heads
    4) Calculate appropriate attention using Q/K/V matrices
    5) Concatenate all of the end matrices together
    """
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

    def _get_config(self):
        """ 
        Get the configuration settings for the 
        Multi-headed attention layer
        """
        
        config = super(MultiHeadAttention, self)._get_config()
        config.update({'num_heads': self.num_heads, 'd_model': self.d_model})
        return config

    def _split_heads(self, inputs, batch_size):
        " Split into the equivalent number of attention heads """
        inputs = tf_reshape(inputs, shape=(batch_size, 
                                           -1, 
                                           self.num_heads, 
                                           self.depth))
        return tf_transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        """ Call each of the key steps """
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf_shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # scaled dot-product attention
        scaled_attention = self._scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf_transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf_reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs
    
    def _scaled_dot_product_attention(self, query, key, value, mask):
        """ Calculate the attention weights """
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


class PositionalEncoding(keras_layers.Layer):
    """ 
    Complete the positional encoding to the embeded words. This allows 
    the Transformer model to determine the relations of the positions
    between the input tokens
    """
        
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_config(self):
        """ get the configuation items """
        config = super(PositionalEncoding, self).get_config()
        config.update({'position': self.position, 'd_model': self.d_model})
        return config

    def get_angles(self, position, i, d_model):
        """ get the appropriate angles for the positional encoding """
        
        angles = 1 / tf_pow(10000, (2 * (i // 2)) / tf_cast(d_model, tf_float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        """ Actually encode the positions to the matrix array """
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
        """ call the position encoding function """
        return inputs + self.pos_encoding[:, :tf_shape(inputs)[1], :]


