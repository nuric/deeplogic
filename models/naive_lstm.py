"""Naive LSTM model."""
from keras.layers import Input, Embedding, LSTM, RepeatVector, Dense
from keras.layers import Bidirectional, concatenate
from keras.models import Model

EMBEDDING_DIM = 25
HIDDEN_DIM = 100

# pylint: disable=line-too-long

def build_model(context_maxlen=60, query_maxlen=10, char_size=27):
  """Build the model."""
  # Inputs
  context = Input(shape=(context_maxlen,), name='context', dtype='int32')
  query = Input(shape=(query_maxlen,), name='query', dtype='int32')

  # Embeddings
  embed = Embedding(char_size, EMBEDDING_DIM, mask_zero=True, name='embed')
  embedded_ctx, embedded_q = embed(context), embed(query)

  # Some feature extraction
  q = LSTM(HIDDEN_DIM, name='query_features')(embedded_q)
  q = RepeatVector(context_maxlen, name='repeat_query')(q)
  ctxq = concatenate([embedded_ctx, q])

  # Prediction
  out = LSTM(HIDDEN_DIM, name='ctxq_read')(ctxq)
  out = Dense(1, activation='sigmoid', name='out')(out)

  model = Model([context, query], out)
  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
  return model
