"""Training module for logic-memnn"""
import argparse
import numpy as np
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, ReduceLROnPlateau

from data_gen import CHAR_IDX
from utils import LogicSeq
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Train logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
parser.add_argument("-d", "--debug", action="store_true", help="Only predict single data point.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+".h5"

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def ask(context, query, model):
  """Predict output for given context and query."""
  rs = context.split('.')[:-1] # split rules
  rs = [r + '.' for r in rs]
  dgen = LogicSeq([(rs, query, 0)], 1, False)
  out = model.predict_generator(dgen)
  return np.asscalar(out)

def train(model, model_file):
  """Train the given model saving weights to model_file."""
  # Setup callbacks
  callbacks = [ModelCheckpoint(filepath=model_file, save_weights_only=True),
               ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.001, verbose=1),
               TerminateOnNaN()]
  # Big data machine learning in the cloud
  traind = LogicSeq.from_file("data/train.txt", 32)
  testd = LogicSeq.from_file("data/test.txt", 32)
  try:
    model.fit_generator(traind, epochs=200,
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

if __name__ == '__main__':
  # Load in the model
  nn_model = build_model(MODEL_NAME, MODEL_FILE,
                         char_size=len(CHAR_IDX)+1)
  nn_model.summary()
  if ARGS.debug:
    print("DEBUG:", ask("p(X):-q(X).q(a).", "p(a).", nn_model))
  else:
    train(nn_model, MODEL_FILE)
