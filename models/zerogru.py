"""ZeroGRU module for nested RNNs."""
import keras
import keras.backend as K
import keras.layers as L

class ZeroGRUCell(L.GRUCell):
  """GRU Cell that skips timestep if inputs are all zero."""
  def call(self, inputs, states, training=None):
    """Step function of the cell."""
    h_tm1 = states[0] # previous output
    cond = K.all(K.equal(inputs, 0), axis=-1, keepdims=True)
    new_output, new_states = super().call(inputs, states, training=training)
    curr_output = K.switch(cond, h_tm1, new_output)
    curr_states = [K.switch(cond, states[i], new_states[i]) for i in range(len(states))]
    return curr_output, curr_states


def ZeroGRU(units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.0,
            recurrent_dropout=0.0,
            implementation=1,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            reset_after=False,
            **kwargs):
    """Layer wrapper for the ZeroGRUCell."""
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
    return L.RNN(cell, return_sequences=return_sequences,
                 return_state=return_state,
                 go_backwards=go_backwards,
                 stateful=stateful,
                 unroll=unroll)

def collapse(inputs):
  """Collapse higher dimensions."""
  shape = K.shape(inputs)
  batch_times_rest = K.prod(shape[:-2], keepdims=True)  # ()
  new_shape = K.concatenate(
      [batch_times_rest, shape[-2:]]
  )  # (B*X, timesteps, features)
  return K.reshape(inputs, new_shape)

def expand(inputs):
  """Expand back into higher dimensions."""
  original_shape = K.shape(inputs[1])  # (B, ..., timesteps, features)
  output_shape = K.shape(inputs[0])  # (B*X, features)
  new_shape = K.concatenate([original_shape[:-2], output_shape[-1:]])
  return K.reshape(inputs[0], new_shape)

collapseL = L.Lambda(collapse, name='collapseL')
expandL = L.Lambda(expand, name='expandL')


def NestedTimeDist(layer, **kwargs):
  """Nested TimeDistributed wrapper for higher rank tensors."""
  def call(inputs, mask=None, training=None, initial_state=None):
    """Wrap reshaping around RNN layer call."""
    nonlocal layer
    collapsed = collapseL(inputs)
    layer_out = layer(collapsed)
    return expandL([layer_out, inputs])

  return call
