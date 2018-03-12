"""Data utils for logic-memnn."""
import numpy as np
from keras.utils import Sequence
from keras.preprocessing.sequence import pad_sequences

from data_gen import CHAR_IDX


class LogicSeq(Sequence):
  """Sequence generator for G-Research data."""
  def __init__(self, data, batch_size, train=True):
    self.data = data or list()
    self.batch_size = batch_size
    self.train = train

  def __len__(self):
    return int(np.ceil(len(self.data) / self.batch_size))

  def __getitem__(self, idx):
    dpoints = self.data[idx*self.batch_size:(idx+1)*self.batch_size]
    # Create batch
    ctxs, queries, targets = list(), list(), list()
    for ctx, q, t in dpoints:
      np.random.shuffle(ctx)
      ctxs.append([CHAR_IDX[c] for c in ''.join(ctx)])
      queries.append([CHAR_IDX[c] for c in q])
      targets.append(int(t))
    xs = [pad_sequences(ctxs, padding='post'),
          pad_sequences(queries, padding='post')]
    if self.train:
      return xs, np.array(targets)
    return xs

  @classmethod
  def from_file(cls, fname, batch_size):
    """Load logic programs from given fname."""
    dpoints = list()
    with open(fname) as f:
      ctx, isnew_ctx = list(), False
      for l in f.readlines():
        l = l.strip()
        if l[0] == '?':
          _, q, t = l.split()
          dpoints.append((ctx.copy(), q, t))
          isnew_ctx = True
        else:
          if isnew_ctx:
            ctx = list()
            isnew_ctx = False
          ctx.append(l)
    print("Example data point from:", fname)
    print(dpoints[0])
    return cls(dpoints, batch_size)
