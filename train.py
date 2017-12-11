"""Training module for logic-memnn"""
import argparse
import random
import numpy as np
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.sequence import pad_sequences

from data_gen import CONST_SYMBOLS, VAR_SYMBOLS, PRED_SYMBOLS, EXTRA_SYMBOLS
import models

# Arguments
parser = argparse.ArgumentParser(description="Train logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
ARGS = parser.parse_args()

CHARS = sorted(list(set(CONST_SYMBOLS+VAR_SYMBOLS+PRED_SYMBOLS+EXTRA_SYMBOLS)))
# Reserve 0 for padding
CHAR_IDX = dict((c, i+1) for i, c in enumerate(CHARS))

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

def vectorise_data(dpoints, char_idx):
  """Return embedding indices of dpoints."""
  ctxs, queries, targets = list(), list(), list()
  for ctx, q, t in dpoints:
    ctxs.append([char_idx[c] for c in ''.join(ctx)])
    queries.append([char_idx[c] for c in q])
    targets.append([int(t)])
  return [pad_sequences(ctxs), pad_sequences(queries)], pad_sequences(targets)

def ask(context, query, model, char_idx):
  """Predict output for given context and query."""
  x, _ = vectorise_data([(context, query, 0)], char_idx)
  return np.asscalar(nn_model.predict(x))

def train(model, model_file, data):
  """Train the given model saving weights to model_file."""
  # Setup callbacks
  callbacks = [ModelCheckpoint(filepath=model_file, save_weights_only=True),
               ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.001, verbose=1),
               TerminateOnNaN()]
  # Big data machine learning in the cloud
  try:
    model.fit(data[0], data[1], batch_size=12,
              epochs=200, callbacks=callbacks)
  finally:
    print("Training terminated.")
    print("OUTPUT:", ask(["p(a)"], "p(a)", model, CHAR_IDX))

if __name__ == '__main__':
  # Load in the model
  nn_model = models.build_model(MODEL_NAME, MODEL_FILE, char_size=len(CHARS)+1)
  nn_model.summary()
  train(nn_model, MODEL_FILE, vectorise_data(load_data("data/task.txt"), CHAR_IDX))
