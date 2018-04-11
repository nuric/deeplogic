"""Evaluation module for logic-memnn"""
import argparse
import numpy as np

from data_gen import CHAR_IDX
from utils import LogicSeq
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Evaluate logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+".h5"

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def evaluate(model):
  """Evaluate model on each test data."""
  for i in range(1, 13):
    dgen = LogicSeq.from_file("data/test_task{}.txt".format(i), 32)
    print(model.evaluate_generator(dgen))

if __name__ == '__main__':
  # Load in the model
  nn_model = build_model(MODEL_NAME, MODEL_FILE,
                         char_size=len(CHAR_IDX)+1,
                         training=True)
  nn_model.summary()
  evaluate(nn_model)
