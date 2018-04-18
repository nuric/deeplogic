"""Naive LSTM model."""
import numpy as np
import keras.layers as L
import keras.backend as K
from keras.models import Model

LATENT_DIM = 64

# pylint: disable=line-too-long

def build_model(char_size=27, iterations=4, training=True):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  var_flat = L.Lambda(lambda x: K.reshape(x, K.stack([-1, K.prod(K.shape(x)[1:])])), name='var_flat')
  flat_ctx = var_flat(context)

  # Onehot embedding
  onehot_weights = np.eye(char_size)
  onehot = L.Embedding(char_size, char_size,
                       trainable=False,
                       weights=[onehot_weights],
                       mask_zero=True,
                       name='onehot')
  embedded_ctx = onehot(flat_ctx) # (?, rules, preds, chars, char_size)
  embedded_q = onehot(query) # (?, chars, char_size)

  # Initial pass
  init_lstm = L.LSTM(LATENT_DIM, return_state=True, name='init_lstm')
  _, *states = init_lstm(embedded_q)
  init_lstm.return_sequences = True
  ctx, *states = init_lstm(embedded_ctx, initial_state=states)

  # Reused layers over iterations
  lstm = L.LSTM(LATENT_DIM, return_sequences=True, return_state=True, name='lstm')

  # Iterations
  for _ in range(iterations):
    ctx, *states = lstm(ctx, initial_state=states)

  # Prediction
  out = L.concatenate(states, name='final_states')
  out = L.Dense(1, activation='sigmoid', name='out')(out)

  model = Model([context, query], out)
  if training:
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
  return model
