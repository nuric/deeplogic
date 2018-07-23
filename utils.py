"""Data utils for logic-memnn."""
import json
import socket
import numpy as np

import keras.callbacks as C
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from data_gen import CHAR_IDX


class LogicSeq(Sequence):
  """Sequence generator for G-Research data."""
  def __init__(self, data, batch_size, train=True, shuffle=True, pad=False):
    self.data = data or list()
    self.batch_size = batch_size
    self.train = train
    self.shuffle = shuffle
    self.pad = pad

  def __len__(self):
    return int(np.ceil(len(self.data) / self.batch_size))

  def on_epoch_end(self):
    """Shuffle data at the end of epoch."""
    if self.shuffle:
      np.random.shuffle(self.data)

  def __getitem__(self, idx):
    dpoints = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
    # Create batch
    ctxs, queries, targets = list(), list(), list()
    for ctx, q, t in dpoints:
      if self.shuffle:
        np.random.shuffle(ctx)
      rules = [r.replace(':-', '.').replace(';', '.').split('.')[:-1]
               for r in ctx]
      if self.pad:
        rules.append(['']) # Append a blank rule
      rules = [[[CHAR_IDX[c] for c in pred]
                for pred in r]
               for r in rules]
      ctxs.append(rules)
      queries.append([CHAR_IDX[c] for c in q[:-1]]) # Remove '.' at the end
      targets.append(t)
    vctxs = np.zeros((len(dpoints),
                      max([len(rs) for rs in ctxs]),
                      max([len(ps) for rs in ctxs for ps in rs]),
                      max([len(cs) for rs in ctxs for ps in rs for cs in ps])),
                     dtype='int')
    # Contexts
    for i in range(len(dpoints)):
      # Rules in context (ie program)
      for j in range(len(ctxs[i])):
        # Predicates in rules
        for k in range(len(ctxs[i][j])):
          # Chars in predicates
          for l in range(len(ctxs[i][j][k])):
            vctxs[i, j, k, l] = ctxs[i][j][k][l]
    xs = [vctxs, pad_sequences(queries, padding='post')]
    if self.train:
      return xs, np.array(targets)
    return xs

  @classmethod
  def from_file(cls, fname, batch_size, pad=False, verbose=True):
    """Load logic programs from given fname."""
    dpoints = list()
    with open(fname) as f:
      ctx, isnew_ctx = list(), False
      for l in f.readlines():
        l = l.strip()
        if l[0] == '?':
          _, q, t = l.split()
          dpoints.append((ctx.copy(), q, int(t)))
          isnew_ctx = True
        else:
          if isnew_ctx:
            ctx = list()
            isnew_ctx = False
          ctx.append(l)
    np.random.shuffle(dpoints)
    if verbose:
      print("Example data points from:", fname)
      print(dpoints[:4])
    return cls(dpoints, batch_size, pad=pad)


class ThresholdStop(C.Callback):
  """Stop when monitored value is greater than threshold."""
  def __init__(self, monitor='val_acc', threshold=0.97):
    super().__init__()
    self.monitor = monitor
    self.threshold = threshold

  def on_epoch_end(self, epoch, logs=None):
    current = logs.get(self.monitor)
    if current >= self.threshold:
      self.model.stop_training = True


class StatefulCheckpoint(C.ModelCheckpoint):
  """Save extra checkpoint data to resume training."""
  def __init__(self, weight_file, state_file=None, **kwargs):
    """Save the state (epoch etc.) along side weights."""
    super().__init__(weight_file, **kwargs)
    self.state_f = state_file
    self.hostname = socket.gethostname()
    self.state = dict()
    if self.state_f:
      # Load the last state if any
      try:
        with open(self.state_f, 'r') as f:
          self.state = json.load(f)
        self.best = self.state['best']
      except Exception as e: # pylint: disable=broad-except
        print("Skipping last state:", e)

  def on_train_begin(self, logs=None):
    prefix = "Resuming" if self.state else "Starting"
    print("{} training on {}".format(prefix, self.hostname))

  def on_epoch_end(self, epoch, logs=None):
    """Saves training state as well as weights."""
    if self.state_f:
      state = {'epoch': epoch+1, 'best': self.best}
      state.update(logs)
      state.update(self.params)
      with open(self.state_f, 'w') as f:
        json.dump(state, f)
    super().on_epoch_end(epoch, logs)

  def get_last_epoch(self, initial_epoch=0):
    """Return last saved epoch if any, or return default argument."""
    return self.state.get('epoch', initial_epoch)

  def on_train_end(self, logs=None):
    print("Training ending on {}".format(self.hostname))
