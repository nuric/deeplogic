"""Training module for logic-memnn"""
import argparse
import random
import numpy as np

# Arguments
parser = argparse.ArgumentParser(description="Train logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+".h5"

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def load_data(fname, shuffle=True):
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

if __name__ == '__main__':
  print(load_data("data/task.txt"))
