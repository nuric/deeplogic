"""Iterative memory attention model."""
import keras.backend as K
import keras.layers as L
from keras.models import Model

PRED_DIM = 50
RULE_DIM = 75
STATE_DIM = 100
ATT_LATENT_DIM = 50
ITERATIONS = 1

# pylint: disable=line-too-long


class NestedTimeDist(L.TimeDistributed):
  """Nested TimeDistributed wrapper for higher rank tensors."""
  # Only override the call to forcefully use the rnn version
  def call(self, inputs, **kwargs):
    def step(x, _):
      output = self.layer.call(x, **kwargs)
      return output, []
    _, outputs, _ = K.rnn(step, inputs,
                          initial_states=[],
                          unroll=False)
    return outputs

def build_model(char_size=27, training=True):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  # Contextual embeddeding of symbols
  onehot = L.Embedding(char_size, char_size,
                       embeddings_initializer='identity',
                       mask_zero=True,
                       trainable=False,
                       name='onehot')
  embedded_ctx = onehot(context) # (?, rules, preds, chars, char_size)
  embedded_q = onehot(query) # (?, chars, char_size)

  # Run the initial pass over the context and query
  embed_pred = L.GRU(PRED_DIM, name='embed_pred')
  embedded_predq = embed_pred(embedded_q) # (?, PRED_DIM)
  embedded_ctx_preds = NestedTimeDist(NestedTimeDist(embed_pred, name='nest1'), name='nest2')(embedded_ctx)
  # (?, rules, preds, PRED_DIM)

  embed_rule = L.GRU(RULE_DIM, go_backwards=True, name='embed_rule')
  embedded_rules = NestedTimeDist(embed_rule, name='d_embed_rule')(embedded_ctx_preds)
  # (?, rules, RULE_DIM)

  # Reused layers over iterations
  rule_to_att = L.TimeDistributed(L.Dense(ATT_LATENT_DIM, name='rule_to_att'), name='d_rule_to_att')
  state_to_att = L.Dense(ATT_LATENT_DIM, name='state_to_att')
  repeat_toctx = L.RepeatVector(K.shape(context)[1], name='repeat_to_ctx')
  att_dense = L.TimeDistributed(L.Dense(1), name='att_dense')
  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')

  unifier = NestedTimeDist(L.GRU(STATE_DIM, name='unifier'), name='dist_unifier')
  gating = L.Dense(1, activation='sigmoid', name='gating')
  gate2 = L.Lambda(lambda xyg: xyg[2]*xyg[0] + (1-xyg[2])*xyg[1], name='gate')

  # Reasoning iterations
  state = L.Dense(STATE_DIM, activation='tanh', name='init_state')(embedded_predq)
  ctx_rules = rule_to_att(embedded_rules) # (?, rules, ATT_LATENT_DIM)
  for _ in range(ITERATIONS):
    # Compute attention between rule and query state
    att_state = state_to_att(state) # (?, ATT_LATENT_DIM)
    att_state = repeat_toctx(att_state) # (?, rules, ATT_LATENT_DIM)
    sim_vec = L.multiply([ctx_rules, att_state])
    sim_vec = att_dense(sim_vec) # (?, rules, 1)
    sim_vec = squeeze2(sim_vec) # (?, rules)
    sim_vec = L.Softmax(axis=1)(sim_vec)

    # Unify every rule and weighted sum based on attention
    new_states = unifier(embedded_ctx_preds, initial_state=[state])
    # (?, rules, STATE_DIM)
    new_state = L.dot([sim_vec, new_states], (1, 1))
    s_m_ns = L.multiply([state, new_state])
    s_s_ns = L.subtract([state, new_state])
    gate = L.concatenate([state, new_state, s_m_ns, s_s_ns])
    gate = gating(gate)
    new_state = gate2([state, new_state, gate])
    state = new_state

  # Predication
  out = L.Dense(1, activation='sigmoid', name='out')(state)
  if training:
    model = Model([context, query], [out])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
  else:
    model = Model([context, query], [sim_vec, out])
  return model
