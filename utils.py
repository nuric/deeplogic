"""Data utils for logic-memnn."""
import numpy as np
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
  def from_file(cls, fname, batch_size, pad=False):
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
    print("Example data points from:", fname)
    print(dpoints[:4])
    return cls(dpoints, batch_size, pad=pad)
