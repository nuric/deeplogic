"""Iterative memory attention model."""
import numpy as np
import keras.backend as K
import keras.layers as L
from keras.models import Model

from .zerogru import ZeroGRU

# pylint: disable=line-too-long

def build_model(char_size=27, dim=64, iterations=4, training=True, ilp=False, pca=False):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='context', dtype='int32')
  query = L.Input(shape=(None,), name='query', dtype='int32')

  # Flatten preds to embed entire rules
  var_flat = L.Lambda(lambda x: K.reshape(x, K.stack([K.shape(x)[0], -1, K.prod(K.shape(x)[2:])])), name='var_flat')
  flat_ctx = var_flat(context) # (?, rules, preds*chars)

  # Onehot embeddeding of symbols
  onehot_weights = np.eye(char_size)
  onehot_weights[0, 0] = 0 # Clear zero index
  onehot = L.Embedding(char_size, char_size,
                       trainable=False,
                       weights=[onehot_weights],
                       name='onehot')
  embedded_ctx = onehot(flat_ctx) # (?, rules, preds*chars, char_size)
  embedded_q = onehot(query) # (?, chars, char_size)

  # Embed predicates
  embed_pred = ZeroGRU(dim, go_backwards=True, return_sequences=True, return_state=True, name='embed_pred')
  embedded_predqs, embedded_predq = embed_pred(embedded_q) # (?, chars, dim)
  embed_rule = ZeroGRU(dim, go_backwards=True, name='embed_rule')
  # Embed every rule
  embedded_rules = L.TimeDistributed(embed_rule, name='rule_embed')(embedded_ctx)
  # (?, rules, dim)

  # Reused layers over iterations
  concatm1 = L.Concatenate(name='concatm1')
  # repeat_toqlen = L.RepeatVector(K.shape(embedded_q)[1], name='repeat_toqlen')
  repeat_toqlen = L.Lambda(lambda xs: K.repeat(xs[0], K.shape(xs[1])[1]), name='repeat_to_qlen')
  mult_cqi = L.Multiply(name='mult_cqi')
  dense_cqi = L.Dense(dim, name='dense_cqi')
  dense_cais = L.Dense(1, name='dense_cais')

  squeeze2 = L.Lambda(lambda x: K.squeeze(x, 2), name='sequeeze2')
  softmax1 = L.Softmax(axis=1, name='softmax1')
  dot11 = L.Dot((1, 1), name='dot11')

  # repeat_toctx = L.RepeatVector(K.shape(context)[1], name='repeat_toctx')
  repeat_toctx = L.Lambda(lambda xs: K.repeat(xs[0], K.shape(xs[1])[1]), name='repeat_to_ctx')
  memory_dense = L.Dense(dim, name='memory_dense')
  kb_dense = L.Dense(dim, name='kb_dense')
  mult_info = L.Multiply(name='mult_info')
  info_dense = L.Dense(dim, name='info_dense')
  mult_att_dense = L.Multiply(name='mult_att_dense')
  read_att_dense = L.Dense(1, name='read_att_dense')

  mem_info_dense = L.Dense(dim, name='mem_info_dense')
  stack1 = L.Lambda(lambda xs: K.stack(xs, 1), output_shape=(None, dim), name='stack1')
  mult_self_att = L.Multiply(name='mult_self_att')
  self_att_dense = L.Dense(1, name='self_att_dense')
  misa_dense = L.Dense(dim, use_bias=False, name='misa_dense')
  mi_info_dense = L.Dense(dim, name='mi_info_dense')
  add_mip = L.Lambda(lambda xy: xy[0]+xy[1], name='add_mip')
  control_gate = L.Dense(1, activation='sigmoid', name='control_gate')
  gate2 = L.Lambda(lambda xyg: xyg[2]*xyg[0] + (1-xyg[2])*xyg[1], name='gate')

  # Init control and memory
  zeros_like = L.Lambda(K.zeros_like, name='zeros_like')
  memory = embedded_predq # (?, dim)
  control = zeros_like(memory) # (?, dim)
  pmemories, pcontrols = [memory], [control]

  # Reasoning iterations
  outs = list()
  for i in range(iterations):
    # Control Unit
    qi = L.Dense(dim, name='qi'+str(i))(embedded_predq) # (?, dim)
    cqi = dense_cqi(concatm1([control, qi])) # (?, dim)
    cais = dense_cais(mult_cqi([repeat_toqlen([cqi, embedded_q]), embedded_predqs])) # (?, qlen, 1)
    cais = squeeze2(cais) # (?, qlen)
    cais = softmax1(cais) # (?, qlen)
    outs.append(cais)
    new_control = dot11([cais, embedded_predqs]) # (?, dim)

    # Read Unit
    info = mult_info([repeat_toctx([memory_dense(memory), context]), kb_dense(embedded_rules)]) # (?, rules, dim)
    infop = info_dense(concatm1([info, embedded_rules])) # (?, rules, dim)
    rai = read_att_dense(mult_att_dense([repeat_toctx([new_control, context]), infop])) # (?, rules, 1)
    rai = squeeze2(rai) # (?, rules)
    rai = softmax1(rai) # (?, rules)
    outs.append(rai)
    read = dot11([rai, embedded_rules]) # (?, dim)

    # Write Unit
    mi_info = mem_info_dense(concatm1([read, memory])) # (?, dim)
    past_ctrls = stack1(pcontrols) # (?, i+1, dim)
    sai = self_att_dense(mult_self_att([L.RepeatVector(i+1)(new_control), past_ctrls])) # (?, i+1, 1)
    sai = squeeze2(sai) # (?, i+1)
    sai = softmax1(sai) # (?, i+1)
    outs.append(sai)
    past_mems = stack1(pmemories) # (?, i+1, dim)
    misa = L.dot([sai, past_mems], (1, 1), name='misa_'+str(i)) # (?, dim)
    mip = add_mip([misa_dense(misa), mi_info_dense(mi_info)]) # (?, dim)
    cip = control_gate(new_control) # (?, 1)
    outs.append(cip)
    new_memory = gate2([mip, memory, cip]) # (?, dim)

    # Update state
    pcontrols = pcontrols + [new_control]
    pmemories = pmemories + [new_memory]
    memory, control = new_memory, new_control

  # Output Unit
  out = L.Dense(1, activation='sigmoid', name='out')(concatm1([embedded_predq, memory]))
  if training:
    model = Model([context, query], out)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
  else:
    model = Model([context, query], outs + [out])
  return model
