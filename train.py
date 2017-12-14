"""Training module for logic-memnn"""
import argparse
import random
import numpy as np
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.sequence import pad_sequences

from data_gen import CONST_SYMBOLS, VAR_SYMBOLS, PRED_SYMBOLS, EXTRA_SYMBOLS
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Train logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
ARGS = parser.parse_args()

CHARS = sorted(list(set(CONST_SYMBOLS+VAR_SYMBOLS+PRED_SYMBOLS+EXTRA_SYMBOLS)))
# Reserve 0 for padding
CHAR_IDX = dict((c, i+1) for i, c in enumerate(CHARS))

# Adjusted after loading data
MAX_CTX_LEN = 40
MAX_Q_LEN = 10

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+".h5"

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def load_data(fname):
  """Load logic programs from given fname."""
  dpoints = list()
  with open(fname) as f:
    ctx, isnew_ctx = list(), False
    for l in f.readlines():
      l = l.strip()
      if l[0] == '?':
        _, q, t = l.split()
        dpoints.append((random.sample(ctx, len(ctx)), q, t))
        isnew_ctx = True
      else:
        if isnew_ctx:
          ctx = list()
          isnew_ctx = False
        ctx.append(l)
  return dpoints

def vectorise_data(dpoints, char_idx, pad=False):
  """Return embedding indices of dpoints."""
  ctxs, queries, targets = list(), list(), list()
  for ctx, q, t in dpoints:
    ctxs.append([char_idx[c] for c in ''.join(ctx)])
    queries.append([char_idx[c] for c in q])
    targets.append(int(t))
  if pad:
    return ([pad_sequences(ctxs, MAX_CTX_LEN),
             pad_sequences(queries, MAX_Q_LEN)],
            np.array(targets))
  return ([pad_sequences(ctxs),
           pad_sequences(queries)],
          np.array(targets))

def ask(context, query, model, char_idx):
  """Predict output for given context and query."""
  x, _ = vectorise_data([(context, query, 0)], char_idx, True)
  return np.asscalar(model.predict(x))

def train(model, model_file, data):
  """Train the given model saving weights to model_file."""
  # Setup callbacks
  callbacks = [ModelCheckpoint(filepath=model_file, save_weights_only=True),
               ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.001, verbose=1),
               TensorBoard(write_images=True, embeddings_freq=1),
               TerminateOnNaN()]
  # Big data machine learning in the cloud
  try:
    model.fit(data[0], data[1], batch_size=12,
              epochs=200, callbacks=callbacks)
  finally:
    print("Training terminated.")
    # Dump some examples for debugging
    samples = [("p(a).", "p(a)"),
               ("p(a).", "p(b)"),
               ("p(X).", "p(c)"),
               ("p(X,Y).", "q(a,b)"),
               ("p(X,X).", "p(a,b)"),
               ("p(X,X).", "p(a,a)")]
    for c, q in samples:
      print("{} ? {} -> {}".format(c, q, ask(c, q, model, CHAR_IDX)))

if __name__ == '__main__':
  # Load the data
  vdata = vectorise_data(load_data("data/task.txt"), CHAR_IDX)
  MAX_CTX_LEN = vdata[0][0].shape[1]
  MAX_Q_LEN = vdata[0][1].shape[1]
  print("MAX_CTX_LEN:", MAX_CTX_LEN)
  print("MAX_Q_LEN:", MAX_Q_LEN)
  # Load in the model
  nn_model = build_model(MODEL_NAME, MODEL_FILE,
                         context_maxlen=MAX_CTX_LEN,
                         query_maxlen=MAX_Q_LEN,
                         char_size=len(CHARS)+1)
  nn_model.summary()
  train(nn_model, MODEL_FILE, vdata)
