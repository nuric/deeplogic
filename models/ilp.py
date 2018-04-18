"""Differentiable ILP."""
import keras.backend as K
import keras.layers as L
from keras.models import Model

import models.ima as ima

# pylint: disable=line-too-long

class RuleTemplate(L.Layer):
  """Layer that exposes trainable weights as rule templates."""
  def __init__(self, goal, num_preds, pred_len, embed_size,
               initializer='random_normal',
               regularizer=None,
               constraint=None,
               **kwargs):
    self.goal = goal
    self.num_preds = num_preds
    self.pred_len = pred_len
    self.embed_size = embed_size
    self.initializer = initializer
    self.regularizer = regularizer
    self.constraint = constraint
    super().__init__(**kwargs)

  def build(self, input_shape):
    """Build the layer."""
    # Build the constant head (goal)
    self.body = self.add_weight(shape=(1, self.num_preds, self.pred_len, self.embed_size),
                                initializer=self.initializer,
                                name='rule',
                                regularizer=self.regularizer,
                                constraint=self.constraint)
    super().build(input_shape)

  def call(self, inputs):
    """Return the rule weights."""
    h = K.constant(self.goal) # (1, 1, pred_len, embed_size)
    b = K.softmax(self.body)
    r = K.concatenate([h, b], axis=1)
    r = K.expand_dims(r, axis=0)
    r = K.tile(r, [K.shape(inputs)[0], 1, 1, 1, 1])
    return r

  def compute_output_shape(self, input_shape):
    """Return output shape."""
    return (input_shape[0], 1, self.num_preds, self.pred_len, self.embed_size)

def build_model(char_size=27, training=True,
                goals=None, num_preds=2, pred_len=6):
  """Build the model."""
  # Inputs
  # Context: (rules, preds, chars,)
  context = L.Input(shape=(None, None, None,), name='subcontext', dtype='int32')
  query = L.Input(shape=(None,), name='outer_query', dtype='int32')

  # Generate templates
  goals = goals or list()
  templates = list()
  for i, g in enumerate(goals):
    t = RuleTemplate(g, num_preds, pred_len, char_size, name='trule'+str(i))(context)
    templates.append(t)

  # Concatenate templates into a single tensor
  if len(templates) >= 2:
    templates = L.concatenate(templates, axis=1)
  else:
    templates = templates[0]
  # (?, num_templates, num_preds, pred_len, char_size)

  # Bind to inference engine model
  auxs, out = ima.build_model(char_size, ilp=[context, query, templates])

  if training:
    model = Model([context, query], [out])
  else:
    model = Model([context, query], [templates] + auxs + [out])
  # Disable training in all other layers
  for l in model.layers:
    if not isinstance(l, RuleTemplate):
      l.trainable = False
  if training:
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
  return model
