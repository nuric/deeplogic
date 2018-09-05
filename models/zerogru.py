"""ZeroGRU module for nested RNNs."""
import keras.backend as K
import keras.layers as L

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

  def compute_mask(self, inputs, mask=None):
    return None
