"""Iterative memory attention model."""
import numpy as np
import keras.backend as K
import keras.layers as L
from keras.models import Model

PRED_DIM = 64
STATE_DIM = 128
ATT_LATENT_DIM = 64
ITERATIONS = 4

# pylint: disable=line-too-long

class ZeroGRUCell(L.GRUCell):
  """GRU Cell that skips timestep if inputs are all zero."""
  def call(self, inputs, states, training=None):
    """Step function of the cell."""
    h_tm1 = states[0] # previous output
    cond = K.all(K.equal(inputs, 0), axis=-1)
    new_output, new_states = super().call(inputs, states, training=training)
    curr_output = K.switch(cond, h_tm1, new_output)
    curr_states = [K.switch(cond, states[i], new_states[i]) for i in range(len(states))]
    return curr_output, curr_states


class ZeroGRU(L.GRU):
  """Layer wrapper for the ZeroGRUCell."""
  def __init__(self, units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
    cell = ZeroGRUCell(units,
                       activation=activation,
                       recurrent_activation=recurrent_activation,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       recurrent_initializer=recurrent_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       recurrent_regularizer=recurrent_regularizer,
                       bias_regularizer=bias_regularizer,
                       kernel_constraint=kernel_constraint,
                       recurrent_constraint=recurrent_constraint,
                       bias_constraint=bias_constraint,
                       dropout=dropout,
                       recurrent_dropout=recurrent_dropout,
                       implementation=implementation,
                       reset_after=reset_after)
    super(L.GRU, self).__init__(cell,
                                return_sequences=return_sequences,
                                return_state=return_state,
                                go_backwards=go_backwards,
                                stateful=stateful,
                                unroll=unroll,
                                **kwargs)
    self.activity_regularizer = L.regularizers.get(activity_regularizer)

class NestedTimeDist(L.TimeDistributed):
  """Nested TimeDistributed wrapper for higher rank tensors."""
  def call(self, inputs, mask=None, training=None, initial_state=None):
    def step(x, _):
      output = self.layer.call(x, mask=mask,
                               training=training,
                               initial_state=initial_state)
      return output, []
    _, outputs, _ = K.rnn(step, inputs,
                          initial_states=[],
                          unroll=False)
    return outputs

def build_model(char_size=27, training=True, ilp=False):
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

  embed_pred = ZeroGRU(PRED_DIM, name='embed_pred')
  embedded_predq = embed_pred(embedded_q) # (?, PRED_DIM)
  # For every rule, for every predicate, embed the predicate
  embedded_ctx_preds = NestedTimeDist(NestedTimeDist(embed_pred, name='nest1'), name='nest2')(embedded_ctx)
  # (?, rules, preds, PRED_DIM)

  # embed_rule = ZeroGRU(PRED_DIM, go_backwards=True, name='embed_rule')
  # embedded_rules = NestedTimeDist(embed_rule, name='d_embed_rule')(embedded_ctx_preds)
  get_heads = L.Lambda(lambda x: x[:, :, 0, :], name='rule_heads')
  embedded_rules = get_heads(embedded_ctx_preds)
  # (?, rules, PRED_DIM)

  # Reused layers over iterations
  rule_to_att = L.TimeDistributed(L.Dense(ATT_LATENT_DIM, name='rule_to_att'), name='d_rule_to_att')
  state_to_att = L.Dense(ATT_LATENT_DIM, name='state_to_att')
  repeat_toctx = L.RepeatVector(K.shape(embedded_ctx)[1], name='repeat_to_ctx')
  att_dense = L.TimeDistributed(L.Dense(1), name='att_dense')
  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')

  unifier = NestedTimeDist(ZeroGRU(STATE_DIM, name='unifier'), name='dist_unifier')
  # gating = L.Dense(1, activation='sigmoid', name='gating')
  # gate2 = L.Lambda(lambda xyg: xyg[2]*xyg[0] + (1-xyg[2])*xyg[1], name='gate')

  # Reasoning iterations
  state = L.Dense(STATE_DIM, activation='tanh', name='init_state')(embedded_predq)
  ctx_rules = rule_to_att(embedded_rules) # (?, rules, ATT_LATENT_DIM)
  outs = list()
  for _ in range(ITERATIONS):
    # Compute attention between rule and query state
    att_state = state_to_att(state) # (?, ATT_LATENT_DIM)
    att_state = repeat_toctx(att_state) # (?, rules, ATT_LATENT_DIM)
    sim_vec = L.multiply([ctx_rules, att_state]) # (?, rules, ATT_LATENT_DIM)
    sim_vec = att_dense(sim_vec) # (?, rules, 1)
    sim_vec = squeeze2(sim_vec) # (?, rules)
    sim_vec = L.Softmax(axis=1)(sim_vec)
    outs.append(sim_vec)

    # Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
    # (?, rules, STATE_DIM)
    state = L.dot([sim_vec, new_states], (1, 1))

  # Predication
  out = L.Dense(1, activation='sigmoid', name='out')(state)
  if ilp:
    return outs, out
  elif training:
    model = Model([context, query], [out])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
  else:
    model = Model([context, query], outs + [out])
  return model
