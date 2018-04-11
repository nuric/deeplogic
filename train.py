"""Training module for logic-memnn"""
import argparse
import numpy as np
import keras.callbacks as C

from data_gen import CHAR_IDX, IDX_CHAR
from utils import LogicSeq
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Train logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
parser.add_argument("-d", "--debug", action="store_true", help="Only predict single data point.")
parser.add_argument("--trainf", default="data/train.txt", help="Training data file.")
parser.add_argument("--testf", default="data/test.txt", help="Testing data file.")
parser.add_argument("-c", "--curriculum", action="store_true", help="Curriculum learning.")
parser.add_argument("-i", "--ilp", action="store_true", help="Run ILP task.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+".h5"

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def ask(context, query, model):
  """Predict output for given context and query."""
  rs = context.split('.')[:-1] # split rules
  rr = [r + '.' for r in rs]
  dgen = LogicSeq([(rr, query, 0)], 1, False, False)
  # print(dgen[0])
  out = model.predict_generator(dgen)
  # print("SHAPES:", [o.shape for o in out])
  for o in out:
    print(o)
  return np.asscalar(out[-1])

def train(model, model_file):
  """Train the given model saving weights to model_file."""
  # Setup callbacks
  callbacks = [C.ModelCheckpoint(filepath=model_file,
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True),
               C.TerminateOnNaN()]
  # Big data machine learning in the cloud
  try:
    if ARGS.curriculum:
      # Train in an incremental fashion
      for i in range(1, 13):
        callbacks = [C.ModelCheckpoint(filepath=model_file,
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True),
                     C.TerminateOnNaN()]
        ft = "data/{}_task1-{}.txt"
        print("ITERATION:", i)
        traind = LogicSeq.from_file(ft.format("train", i), 32)
        testd = LogicSeq.from_file(ft.format("test", i), 32)
        model.fit_generator(traind, epochs=i*3,
                            callbacks=callbacks,
                            validation_data=testd,
                            shuffle=True)
    else:
      traind = LogicSeq.from_file(ARGS.trainf, 32)
      testd = LogicSeq.from_file(ARGS.testf, 32)
      model.fit_generator(traind, epochs=120,
                          callbacks=callbacks,
                          validation_data=testd,
                          shuffle=True)
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
  traind = LogicSeq.from_file("data/ilp_train.txt", 32)
  testd = LogicSeq.from_file("data/ilp_test.txt", 32)
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


def debug(model):
  """Run a single data point for debugging."""
  ctx = "q(a).p(Y):-r(Y).r(X):-q(X).r(a)."
  q = "p(a)."
  # ctx = "r(X):-t(X).t(a).x(Y):-r(Y).t(Y):-p(Y).p(b)."
  # q = "x(b)."
  # ctx = "k(w).a(O):-o(O).o(C):-k(C).p(v)."
  # q = "p(b)."
  # ctx = "r(a,b).q(Y,X):-r(X,Y).p(Z,T):-q(Z,T)."
  # q = "p(b,a)."
  # ctx = "l(t,s).l(X,Y):-l(X,Z);l(Z,Y).l(s,a)."
  # q = "l(t,b)."
  # with open("/homes/nuric/ntp/data/countries/countries_S1.nl") as f:
    # ctx = "".join([l.strip() for l in f if '(nor' in l])
  # ctx += "l(X,Y):-l(X,Z);l(Z,Y)."
  # ctx += "l(neu,eu)."
  # q = "l(nor,eu)."
  print("CTX:", ctx)
  print("Q:", q)
  print("OUT:", ask(ctx, q, model))

if __name__ == '__main__':
  if ARGS.ilp:
    ilp(not ARGS.debug)
  else:
    # Load in the model
    nn_model = build_model(MODEL_NAME, MODEL_FILE,
                           char_size=len(CHAR_IDX)+1,
                           training=not ARGS.debug)
    nn_model.summary()
    if ARGS.debug:
      debug(nn_model)
    else:
      train(nn_model, MODEL_FILE)
