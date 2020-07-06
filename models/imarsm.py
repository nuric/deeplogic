"""Iterative memory attention model."""
import numpy as np
import keras.backend as K
import keras.layers as L
from keras.models import Model

from .zerogru import ZeroGRU, NestedTimeDist

# pylint: disable=line-too-long

def build_model(char_size=27, dim=64, iterations=4, training=True, ilp=False, pca=False):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  if ilp:
    context, query, templates = ilp

  # Contextual embeddeding of symbols
  onehot_weights = np.eye(char_size)
  onehot_weights[0, 0] = 0 # Clear zero index
  onehot = L.Embedding(char_size, char_size,
                       trainable=False,
                       weights=[onehot_weights],
                       name='onehot')
  embedded_ctx = onehot(context) # (?, rules, preds, chars, char_size)
  embedded_q = onehot(query) # (?, chars, char_size)

  if ilp:
    # Combine the templates with the context, (?, rules+temps, preds, chars, char_size)
    embedded_ctx = L.Lambda(lambda xs: K.concatenate(xs, axis=1), name='template_concat')([templates, embedded_ctx])
    # embedded_ctx = L.concatenate([templates, embedded_ctx], axis=1)

  embed_pred = ZeroGRU(dim, go_backwards=True, name='embed_pred')
  embedded_predq = embed_pred(embedded_q) # (?, dim)
  # For every rule, for every predicate, embed the predicate
  embedded_ctx_preds = NestedTimeDist(NestedTimeDist(embed_pred, name='nest1'), name='nest2')(embedded_ctx)
  # (?, rules, preds, dim)

  embed_rule = ZeroGRU(dim, name='embed_rule')
  embedded_rules = NestedTimeDist(embed_rule, name='d_embed_rule')(embedded_ctx_preds)
  # (?, rules, dim)

  # Reused layers over iterations
  repeat_toctx = L.Lambda(lambda xs: K.repeat(xs[0], K.shape(xs[1])[1]), name='repeat_to_ctx')
  diff_sq = L.Lambda(lambda xy: K.square(xy[0]-xy[1]), output_shape=(None, dim), name='diff_sq')
  mult = L.Multiply()
  concat = L.Lambda(lambda xs: K.concatenate(xs, axis=2), output_shape=(None, dim*5), name='concat')
  att_densel = L.Dense(dim//2, activation='tanh', name='att_densel')
  att_dense = L.Dense(1, name='att_dense')
  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')
  softmax1 = L.Softmax(axis=1)
  unifier = NestedTimeDist(ZeroGRU(dim, go_backwards=False, name='unifier'), name='dist_unifier')
  dot11 = L.Dot((1, 1))

  # Reasoning iterations
  state = embedded_predq
  repeated_q = repeat_toctx([embedded_predq, embedded_ctx])
  outs = list()
  for _ in range(iterations):
    # Compute attention between rule and query state
    ctx_state = repeat_toctx([state, embedded_ctx]) # (?, rules, dim)
    s_s_c = diff_sq([ctx_state, embedded_rules])
    s_m_c = mult([embedded_rules, state]) # (?, rules, dim)
    sim_vec = concat([s_s_c, s_m_c, ctx_state, embedded_rules, repeated_q])
    sim_vec = att_densel(sim_vec) # (?, rules, dim//2)
    sim_vec = att_dense(sim_vec) # (?, rules, 1)
    sim_vec = squeeze2(sim_vec) # (?, rules)
    sim_vec = softmax1(sim_vec)
    outs.append(sim_vec)

    # Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
    # (?, rules, dim)
    state = dot11([sim_vec, new_states])

  # Predication
  out = L.Dense(1, activation='sigmoid', name='out')(state)
  if ilp:
    return outs, out
  elif pca:
    model = Model([context, query], [embedded_rules])
  elif training:
    model = Model([context, query], [out])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
  else:
    model = Model([context, query], outs + [out])
  return model
