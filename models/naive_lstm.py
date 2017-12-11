"""Naive LSTM model."""
from keras.layers import Input, Embedding, LSTM, Dense
from keras.layers import Bidirectional, concatenate
from keras.models import Model

EMBEDDING_DIM = 50
HIDDEN_DIM = 150

# pylint: disable=line-too-long

def build_model(char_size=27):
  """Build the model."""
  # Inputs
  context = Input(shape=(None,), name='context', dtype='int32')
  query = Input(shape=(None,), name='query', dtype='int32')

  # Embeddings
  embedded_ctx = Embedding(char_size, EMBEDDING_DIM, mask_zero=True, name='ctx_embed')(context)
  embedded_q = Embedding(char_size, EMBEDDING_DIM, mask_zero=True, name='query_embed')(query)

  # Some feature extraction
  ctx = Bidirectional(LSTM(HIDDEN_DIM), name='ctx_features')(embedded_ctx)
  q = LSTM(HIDDEN_DIM, name='query_features')(embedded_q)
  ctxq = concatenate([ctx, q])

  # Prediction
  out = Dense(1, activation='sigmoid', name='out')(ctxq)

  model = Model([context, query], out)
  model.compile(loss='binary_crossentropy',
                optimizer='sgd',
                metrics=['acc'])
  return model
