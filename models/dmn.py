"""Vanilla Dynamic Memory Network."""
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.layers as L
from keras.models import Model

from .zerogru import ZeroGRU, NestedTimeDist

# pylint: disable=line-too-long


class EpisodicMemory(L.Wrapper):
  """Episodic memory from DMN."""
  def __init__(self, units, **kwargs):
    self.grucell = L.GRUCell(units, name=kwargs['name']+'_gru') # Internal cell
    super().__init__(self.grucell, **kwargs)

  def build(self, input_shape):
    """Build the layer."""
    _, _, ctx_shape = input_shape
    self.grucell.build((ctx_shape[0],) + ctx_shape[2:])
    super().build(input_shape)

  def call(self, inputs):
    """Compute new state episode."""
    init_state, atts, cs = inputs
    # GRU pass over the facts, according to the attention mask.
    while_valid_index = lambda state, index: index < tf.shape(cs)[1]
    retain = 1 - atts
    update_state = (lambda state, index: (atts[:,index,:] * self.grucell.call(cs[:,index,:], [state])[0] + retain[:,index,:] * state))
    # Loop over context
    final_state, _ = tf.while_loop(while_valid_index,
                     (lambda state, index: (update_state(state, index), index+1)),
                     loop_vars = [init_state, 0])
    return final_state

  def compute_output_shape(self, input_shape):
    """Collapse time dimension."""
    return input_shape[0]

def build_model(char_size=27, dim=64, iterations=4, training=True, pca=False):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  # Flatten preds to embed entire rules
  var_flat = L.Lambda(lambda x: K.reshape(x, K.stack([K.shape(x)[0], -1, K.prod(K.shape(x)[2:])])), name='var_flat')
  flat_ctx = var_flat(context) # (?, rules, preds*chars)

  # Onehot embedding
  # Contextual embeddeding of symbols
  onehot_weights = np.eye(char_size)
  onehot_weights[0, 0] = 0 # Clear zero index
  onehot = L.Embedding(char_size, char_size,
                       trainable=False,
                       weights=[onehot_weights],
                       name='onehot')
  embedded_ctx = onehot(flat_ctx) # (?, rules, preds*chars*char_size)
  embedded_q = onehot(query) # (?, chars, char_size)

  embed_pred = ZeroGRU(dim, go_backwards=True, name='embed_pred')
  embedded_predq = embed_pred(embedded_q) # (?, dim)
  # Embed every rule
  embedded_rules = NestedTimeDist(embed_pred, name='rule_embed')(embedded_ctx)
  # (?, rules, dim)

  # Reused layers over iterations
  repeat_toctx = L.RepeatVector(K.shape(embedded_ctx)[1], name='repeat_to_ctx')
  diff_sq = L.Lambda(lambda xy: K.square(xy[0]-xy[1]), output_shape=(None, dim), name='diff_sq')
  concat = L.Lambda(lambda xs: K.concatenate(xs, axis=2), output_shape=(None, dim*5), name='concat')
  att_dense1 = L.TimeDistributed(L.Dense(dim, activation='tanh', name='att_dense1'), name='d_att_dense1')
  att_dense2 = L.TimeDistributed(L.Dense(1, activation='sigmoid', name='att_dense2'), name='d_att_dense2')
  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')
  # expand = L.Lambda(lambda x: K.expand_dims(x, axis=2), name='expand')
  rule_mask = L.Lambda(lambda x: K.cast(K.any(K.not_equal(x, 0), axis=-1, keepdims=True), 'float32'), name='rule_mask')(embedded_rules)
  episodic_mem = EpisodicMemory(dim, name='episodic_mem')

  # Reasoning iterations
  state = embedded_predq
  repeated_q = repeat_toctx(embedded_predq)
  outs = list()
  for _ in range(iterations):
    # Compute attention between rule and query state
    ctx_state = repeat_toctx(state) # (?, rules, dim)
    s_s_c = diff_sq([ctx_state, embedded_rules])
    s_m_c = L.multiply([embedded_rules, state]) # (?, rules, dim)
    sim_vec = concat([s_s_c, s_m_c, ctx_state, embedded_rules, repeated_q])
    sim_vec = att_dense1(sim_vec) # (?, rules, dim)
    sim_vec = att_dense2(sim_vec) # (?, rules, 1)
    # sim_vec = squeeze2(sim_vec) # (?, rules)
    # sim_vec = L.Softmax(axis=1)(sim_vec)
    # sim_vec = expand(sim_vec) # (?, rules, 1)
    sim_vec = L.multiply([sim_vec, rule_mask])

    state = episodic_mem([state, sim_vec, embedded_rules])
    sim_vec = squeeze2(sim_vec) # (?, rules)
    outs.append(sim_vec)

  # Predication
  out = L.Dense(1, activation='sigmoid', name='out')(state)
  if pca:
    model = Model([context, query], [embedded_rules])
  elif training:
    model = Model([context, query], [out])
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
  else:
    model = Model([context, query], outs + [out])
  return model
