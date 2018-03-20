"""Iterative cross attention model."""
import keras.backend as K
from keras.layers import Input, Activation, Dense, GRU, Embedding, RepeatVector, Permute, Flatten
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.merge import dot, concatenate
from keras.models import Model

LATENT_DIM = 50
ITERATIONS = 1

# pylint: disable=line-too-long

def build_model(char_size=27):
  """Build the model."""
  # Inputs
  context = Input(shape=(None,), name='context', dtype='int32')
  query = Input(shape=(None,), name='query', dtype='int32')

  # Contextual embeddeding of symbols
  onehot = Embedding(char_size, char_size,
                     embeddings_initializer='identity',
                     mask_zero=True,
                     trainable=False,
                     name='onehot')
  embedded_ctx = onehot(context)
  embedded_q = onehot(query)

  # Reused layers over iterations
  rembed_ctx = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='re_embed_ctx')
  rembed_q = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='re_embed_query')
  update_q = Bidirectional(GRU(LATENT_DIM, return_sequences=True), name='update_query')
  decode_q = TimeDistributed(Dense(char_size, activation='softmax'), name='decode_query')

  rembedded_ctx = rembed_ctx(embedded_ctx)
  embedded_qq = embedded_q
  for _ in range(ITERATIONS):
    rembedded_q = rembed_q(embedded_qq)
    # (samples, context_maxlen, 2*LATENT_DIM), (samples, query_maxlen, 2*LATENT_DIM)

    # Cross attention mechanism based on similarity
    match = dot([rembedded_ctx, rembedded_q], axes=(2, 2), name='similarity')
    # (samples, context_maxlen, query_maxlen)
    ctxq_match = Activation('softmax')(match)
    qctx_match = Permute((2, 1))(match) # (samples, query_maxlen, context_maxlen)
    qctx_match = Activation('softmax')(qctx_match)

    # Weighted sum using attention
    attended_ctx = dot([qctx_match, rembedded_ctx], axes=(2, 1)) # (samples, query_maxlen, 2*LATENT_DIM)
    merged_q = concatenate([embedded_q, rembedded_q, attended_ctx], axis=2)
    # (samples, query_maxlen, 5*LATENT_DIM)

    # Update context and query
    new_q = update_q(merged_q) # (samples, query_maxlen, 2*LATENT_DIM)
    embedded_qq = decode_q(new_q) # (samples, query_maxlen, char_size)

  # Predication
  final_q = GRU(LATENT_DIM, name='final_query')(embedded_qq)
  out = Dense(1, activation='sigmoid', use_bias=False, name='out')(final_q)

  model = Model([context, query], out)
  model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])
  return model
