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
parser.add_argument("-mf", "--model_file", help="Model filename.")
parser.add_argument("-md", "--model_dir", help="Model weights directory ending with /.")
parser.add_argument("--dim", default=64, type=int, help="Latent dimension.")
parser.add_argument("-d", "--debug", action="store_true", help="Only predict single data point.")
parser.add_argument("--trainf", default="data/train.txt", help="Training data file.")
parser.add_argument("--testf", default="data/test.txt", help="Testing data file.")
parser.add_argument("-s", "--summary", action="store_true", help="Dump model summary on creation.")
parser.add_argument("-c", "--curriculum", action="store_true", help="Curriculum learning.")
parser.add_argument("-ic", "--iter_curriculum", action="store_true", help="Iterative curriculum learning.")
parser.add_argument("-i", "--ilp", action="store_true", help="Run ILP task.")
parser.add_argument("-its", "--iterations", default=4, type=int, help="Number of model iterations.")
parser.add_argument("-bs", "--batch_size", default=32, type=int, help="Training batch_size.")
parser.add_argument("-p", "--pad", action="store_true", help="Pad context with blank rule.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FNAME = ("curr_" if ARGS.curriculum else "multi_") + MODEL_NAME + str(ARGS.dim)
MODEL_FNAME = ARGS.model_file or MODEL_FNAME
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
  dgen = LogicSeq([(rr, query, 0)], 1, False, False, pad=ARGS.pad)
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
               C.EarlyStopping(patience=10, verbose=1),
               C.TerminateOnNaN()]
  # Big data machine learning in the cloud
  try:
    if ARGS.curriculum:
      # Train in an incremental fashion
      for i, its in zip(range(1, 6), [1, 1, 2, 3, 4]):
        print("TASK:", i, "ITERATIONS:", its)
        model = create_model(iterations=its)
        callbacks[0].best = np.Inf # Reset checkpoint
        ft = "data/{}_task1-{}.txt"
        traind = LogicSeq.from_file(ft.format("train", i), ARGS.batch_size, pad=ARGS.pad)
        testd = LogicSeq.from_file(ft.format("test", i), ARGS.batch_size, pad=ARGS.pad)
        model.fit_generator(traind, epochs=i*2,
                            callbacks=callbacks,
                            validation_data=testd,
                            verbose=2,
                            shuffle=True)
    elif ARGS.iter_curriculum:
      # Train incrementally based on iteration count
      for i in range(1, 5):
        print("ITER:", i)
        model = create_model(iterations=i)
        callbacks[0].best = np.Inf # Reset checkpoint
        ft = "data/{}_iter{}.txt"
        traind = LogicSeq.from_file(ft.format("train", i), ARGS.batch_size, pad=ARGS.pad)
        testd = LogicSeq.from_file(ft.format("test", i), ARGS.batch_size, pad=ARGS.pad)
        model.fit_generator(traind, epochs=i*10,
                            callbacks=callbacks,
                            validation_data=testd,
                            verbose=2,
                            shuffle=True)
    # Run full training
    model = create_model(iterations=ARGS.iterations)
    # For long running training swap in stateful checkpoint
    callbacks[0] = StatefulCheckpoint(MODEL_WF, MODEL_SF,
                                      verbose=1, save_best_only=True,
                                      save_weights_only=True)
    traind = LogicSeq.from_file(ARGS.trainf, ARGS.batch_size, pad=ARGS.pad)
    testd = LogicSeq.from_file(ARGS.testf, ARGS.batch_size, pad=ARGS.pad)
    model.fit_generator(traind, epochs=120,
                        callbacks=callbacks,
                        validation_data=testd,
                        verbose=2, shuffle=True,
                        initial_epoch=callbacks[0].get_last_epoch())
  finally:
    print("Training terminated.")
    # Dump some examples for debugging
    samples = [("p(a).", "p(a)."),
               ("p(a).", "p(b)."),
               ("p(X).", "p(c)."),
               ("p(X,Y).", "q(a,b)."),
               ("p(X,X).", "p(a,b)."),
               ("p(X,X).", "p(a,a)."),
               ("p(X):-q(X).r(a).", "p(a).")]
    for c, q in samples:
      print("{} ? {} -> {}".format(c, q, ask(c, q, model)))

def debug():
  """Run a single data point for debugging."""
  # Add command line history support
  import readline # pylint: disable=unused-variable
  model = create_model(iterations=ARGS.iterations, training=False)
  while True:
    try:
      ctx = input("CTX: ").replace(' ', '')
      if not ctx:
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
    model.fit_generator(traind, epochs=200,
                        callbacks=callbacks,
                        validation_data=testd,
                        shuffle=True)
  else:
    # Dummy input to get templates
    ctx = "b(h).v(O):-c(O).c(a)."
    ctx = ctx.split('.')[:-1] # split rules
    ctx = [r + '.' for r in ctx]
    dgen = LogicSeq([(ctx, "f(h).", 0)], 1, False, False)
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
