"""Iterative cross attention model."""
import keras.backend as K
from keras.layers import Input, Activation, Dense, GRU, Lambda, Permute, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import dot, concatenate
from keras.models import Model

LATENT_DIM = 50
ITERATIONS = 1

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

  # Reused layers over iterations
  rembed = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='re_embed')
  update_ctx = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='update_ctx')
  update_q = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='update_query')
  decode_ctx = TimeDistributed(Dense(char_size, activation='softmax', use_bias=False), name='decode_ctx')
  decode_q = TimeDistributed(Dense(char_size, activation='softmax', use_bias=False), name='decode_query')

  for _ in range(ITERATIONS):
    rembedded_ctx, rembedded_q = rembed(embedded_ctx), rembed(embedded_q)
    # (samples, context_maxlen, 2*LATENT_DIM), (samples, query_maxlen, 2*LATENT_DIM)

    # Cross attention mechanism based on similarity
    match = dot([rembedded_ctx, rembedded_q], axes=(2, 2), normalize=True) # (samples, context_maxlen, query_maxlen)
    ctxq_match = Activation('softmax')(match)
    qctx_match = Permute((2, 1))(match) # (samples, query_maxlen, context_maxlen)
    qctx_match = Activation('softmax')(qctx_match)

    # Weighted sum using attention
    attended_query = dot([ctxq_match, rembedded_q], axes=(2, 1)) # (samples, context_maxlen, 2* LATENT_DIM)
    merged_ctx = concatenate([rembedded_ctx, attended_query], axis=2) # (samples, context_maxlen, 4*LATENT_DIM)
    attended_ctx = dot([qctx_match, rembedded_ctx], axes=(2, 1)) # (samples, query_maxlen, 2*LATENT_DIM)
    merged_q = concatenate([rembedded_q, attended_ctx], axis=2) # (samples, query_maxlen, 4*LATENT_DIM)

    # Update context and query
    new_ctx = update_ctx(merged_ctx) # (samples, context_maxlen, 2*LATENT_DIM)
    embedded_ctx = decode_ctx(new_ctx) # (samples, context_maxlen, char_size)
    new_q = update_q(merged_q) # (samples, query_maxlen, 2*LATENT_DIM)
    embedded_q = decode_q(new_q) # (samples, query_maxlen, char_size)

  # Predication
  final_q = Flatten(name='final_query')(embedded_q)
  out = Dense(1, activation='sigmoid', use_bias=False, name='out')(final_q)

  model = Model([context, query], out)
  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
  return model
