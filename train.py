"""Training module for logic-memnn"""
import argparse
import numpy as np
import keras.callbacks as C

from data_gen import CHAR_IDX, IDX_CHAR
from utils import LogicSeq, StatefulCheckpoint, ThresholdStop
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Train logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
parser.add_argument("model_file", help="Model filename.")
parser.add_argument("-md", "--model_dir", help="Model weights directory ending with /.")
parser.add_argument("--dim", default=64, type=int, help="Latent dimension.")
parser.add_argument("-d", "--debug", action="store_true", help="Only predict single data point.")
parser.add_argument("-ts", "--tasks", nargs='*', type=int, help="Tasks to train on, blank for all tasks.")
parser.add_argument("-e", "--epochs", default=120, type=int, help="Number of epochs to train.")
parser.add_argument("-s", "--summary", action="store_true", help="Dump model summary on creation.")
parser.add_argument("-i", "--ilp", action="store_true", help="Run ILP task.")
parser.add_argument("-its", "--iterations", default=4, type=int, help="Number of model iterations.")
parser.add_argument("-bs", "--batch_size", default=32, type=int, help="Training batch_size.")
parser.add_argument("-p", "--pad", action="store_true", help="Pad context with blank rule.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FNAME = ARGS.model_file
MODEL_WF = (ARGS.model_dir or "weights/") + MODEL_FNAME + '.h5'
MODEL_SF = (ARGS.model_dir or "weights/") + MODEL_FNAME + '.json'

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def create_model(**kwargs):
  """Create model from global arguments."""
  # Load in the model
  model = build_model(MODEL_NAME, MODEL_WF,
                      char_size=len(CHAR_IDX)+1,
                      dim=ARGS.dim,
                      **kwargs)
  if ARGS.summary:
    model.summary()
  return model

def ask(context, query, model):
  """Predict output for given context and query."""
  rs = context.split('.')[:-1] # split rules
  rr = [r + '.' for r in rs]
  dgen = LogicSeq([[(rr, query, 0)]], 1, False, False, pad=ARGS.pad)
  # print(dgen[0])
  out = model.predict_generator(dgen)
  # print("SHAPES:", [o.shape for o in out])
  for o in out:
    print(o)
  return np.asscalar(out[-1])

def train():
  """Train the given model saving weights to model_file."""
  # Setup callbacks
  callbacks = [C.ModelCheckpoint(filepath=MODEL_WF,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True),
               ThresholdStop(),
               C.EarlyStopping(monitor='loss', patience=10, verbose=1),
               C.TerminateOnNaN()]
  # Big data machine learning in the cloud
  ft = "data/{}_task{}.txt"
  model = create_model(iterations=ARGS.iterations)
  # For long running training swap in stateful checkpoint
  callbacks[0] = StatefulCheckpoint(MODEL_WF, MODEL_SF,
                                    verbose=1, save_best_only=True,
                                    save_weights_only=True)
  tasks = ARGS.tasks or range(1, 13)
  traind = LogicSeq.from_files([ft.format("train", i) for i in tasks], ARGS.batch_size, pad=ARGS.pad)
  vald = LogicSeq.from_files([ft.format("val", i) for i in tasks], ARGS.batch_size, pad=ARGS.pad)
  model.fit(traind, epochs=ARGS.epochs,
            callbacks=callbacks,
            validation_data=vald,
            verbose=1, shuffle=True,
            initial_epoch=callbacks[0].get_last_epoch())

def debug():
  """Run a single data point for debugging."""
  # Add command line history support
  import readline # pylint: disable=unused-variable
  model = create_model(iterations=ARGS.iterations, training=False)
  while True:
    try:
      ctx = input("CTX: ").replace(' ', '')
      if ctx == 'q':
        break
      q = input("Q: ").replace(' ', '')
      print("OUT:", ask(ctx, q, model))
    except(KeyboardInterrupt, EOFError, SystemExit):
      break
  print("\nTerminating.")


def ilp(training=True):
  """Run the ILP task using the ILP model."""
  # Create the head goal
  goals, vgoals = ["f(X)"], list()
  for g in goals:
    v = np.zeros((1, 1, 4, len(CHAR_IDX)+1))
    for i, c in enumerate(g):
      v[0, 0, i, CHAR_IDX[c]] = 1
    vgoals.append(v)
  # Create the ILP wrapper model
  model = build_model("ilp", "weights/ilp.h5",
                      char_size=len(CHAR_IDX)+1,
                      training=training,
                      goals=vgoals,
                      num_preds=1,
                      pred_len=4)
  model.summary()
  traind = LogicSeq.from_file("data/ilp_train.txt", ARGS.batch_size, pad=ARGS.pad)
  testd = LogicSeq.from_file("data/ilp_test.txt", ARGS.batch_size, pad=ARGS.pad)
  if training:
    # Setup callbacks
    callbacks = [C.ModelCheckpoint(filepath="weights/ilp.h5",
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True),
                 C.TerminateOnNaN()]
    model.fit(traind, epochs=200,
              callbacks=callbacks,
              validation_data=testd,
              shuffle=True)
  else:
    # Dummy input to get templates
    ctx = "b(h).v(O):-c(O).c(a)."
    ctx = ctx.split('.')[:-1] # split rules
    ctx = [r + '.' for r in ctx]
    dgen = LogicSeq([[(ctx, "f(h).", 0)]], 1, False, False)
    print("TEMPLATES:")
    outs = model.predict_on_batch(dgen[0])
    ts, out = outs[0], outs[-1]
    print(ts)
    # Decode template
    # (num_templates, num_preds, pred_length, char_size)
    ts = np.argmax(ts[0], axis=-1)
    ts = np.vectorize(lambda i: IDX_CHAR[i])(ts)
    print(ts)
    print("CTX:", ctx)
    for o in outs[1:-1]:
      print(o)
    print("OUT:", out)

if __name__ == '__main__':
  if ARGS.ilp:
    ilp(not ARGS.debug)
  elif ARGS.debug:
    debug()
  else:
    train()
