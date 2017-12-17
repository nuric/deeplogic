"""Iterative cross attention model."""
import keras.backend as K
from keras.layers import Input, Activation, Dense, GRU, Lambda, Permute
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import dot, concatenate
from keras.models import Model

LATENT_DIM = 75
ITERATIONS = 2

# pylint: disable=line-too-long

def build_model(context_maxlen=60, query_maxlen=10, char_size=27):
  """Build the model."""
  # Inputs
  context = Input(shape=(context_maxlen,), name='context', dtype='int32')
  query = Input(shape=(query_maxlen,), name='query', dtype='int32')

  # Contextual embeddeding of symbols
  embedded_ctx = Lambda(K.one_hot, arguments={'num_classes':char_size},
                        output_shape=(context_maxlen, char_size), name='embed_context')(context)
  embedded_q = Lambda(K.one_hot, arguments={'num_classes':char_size},
                      output_shape=(query_maxlen, char_size), name='embed_query')(query)

  rembed = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='re_embed')
  rembedded_ctx, rembedded_q = rembed(embedded_ctx), rembed(embedded_q)
  # (samples, context_maxlen, 2*LATENT_DIM), (samples, query_maxlen, 2*LATENT_DIM)

  # Cross attention mechanism based on similarity
  match = dot([rembedded_ctx, rembedded_q], axes=(2, 2)) # (samples, context_maxlen, query_maxlen)
  ctxq_match = Activation('softmax')(match)
  qctx_match = Permute((2, 1))(match) # (samples, query_maxlen, context_maxlen)
  qctx_match = Activation('softmax')(qctx_match)

  # Weighted sum using attention
  attended_query = dot([ctxq_match, rembedded_q], axes=(2, 1)) # (samples, context_maxlen, 2* LATENT_DIM)
  merged_ctx = concatenate([rembedded_ctx, attended_query], axis=2) # (samples, context_maxlen, 4*LATENT_DIM)
  attended_ctx = dot([qctx_match, rembedded_ctx], axes=(2, 1)) # (samples, query_maxlen, 2*LATENT_DIM)
  merged_q = concatenate([rembedded_q, attended_ctx], axis=2) # (samples, query_maxlen, 4*LATENT_DIM)

  # Final read over updated memory
  model_ctx = Bidirectional(GRU(LATENT_DIM), name='model_context')(merged_ctx) # (samples, 2*LATENT_DIM)
  model_q = Bidirectional(GRU(LATENT_DIM), name='model_query')(merged_q) # (samples, 2*LATENT_DIM)

  # Predication
  model_combined = concatenate([model_ctx, model_q])
  out = Dense(1, activation='sigmoid', name='out')(model_combined)

  model = Model([context, query], out)
  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
  return model
