"""Evaluation module for logic-memnn"""
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg") # Bypass X server
import matplotlib.pyplot as plt

from data_gen import CHAR_IDX
from utils import LogicSeq
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Evaluate logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
parser.add_argument("-f", "--function", default="evaluate", help="Function to run.")
parser.add_argument("--outf", default="plot.png", help="Plot output file.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+".h5"

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def evaluate():
  """Evaluate model on each test data."""
  model = build_model(MODEL_NAME, MODEL_FILE,
                      char_size=len(CHAR_IDX)+1,
                      training=True)
  model.summary()
  for i in range(1, 13):
    dgen = LogicSeq.from_file("data/test_task{}.txt".format(i), 32)
    print(model.evaluate_generator(dgen))

def get_pca(context, model):
  """Plot the PCA of predicate embeddings."""
  dgen = LogicSeq([(context, "z(z).", 0)], 1, False, False)
  embds = model.predict_generator(dgen)
  embds = embds.squeeze()
  pca = PCA(2)
  embds = pca.fit_transform(embds)
  print("TRANSFORMED:", embds)
  print("VAR:", pca.explained_variance_ratio_)
  return embds

def plot_single_preds():
  """Plot embeddings of single character predicates."""
  model = build_model(MODEL_NAME, MODEL_FILE,
                      char_size=len(CHAR_IDX)+1,
                      pca=True)
  model.summary()
  syms = "abcdefghijklmnopqrstuvwxyz"
  ctx, splits, colours = list(), list(), list()
  # plt.figure(figsize=(8,8))
  for p in ['p', 'q', 'd']:
    col = np.random.rand(3)
    for c in syms:
      colours.append(col)
      ctx.append("{}({}).".format(p,c))
    splits.append(len(ctx))
  for c in ['a', 'y', 'd']:
    col = np.random.rand(3)
    for p in syms:
      colours.append(col)
      ctx.append("{}({}).".format(p,c))
    splits.append(len(ctx))
  embds = get_pca(ctx, model)
  prev_sp = 0
  for sp in splits:
    plt.scatter(embds[prev_sp:sp, 0], embds[prev_sp:sp, 1])
    prev_sp = sp
  for pred, x, y in zip(ctx, embds[:, 0], embds[:, 1]):
    if np.random.rand() < 0.1:
      plt.annotate(pred, xy=(x, y), xytext=(0, 10), textcoords='offset points')
  # plt.axis('scaled')
  plt.title("Single Character Predicates")
  plt.legend(["p(?).", "q(?).", "d(?).", "?(a).", "?(y).", "?(d)."])
  plt.savefig(ARGS.outf, bbox_inches='tight')

def plot_custom_preds():
  """Plot embeddings of custom predicates."""
  model = build_model(MODEL_NAME, MODEL_FILE,
                      char_size=len(CHAR_IDX)+1,
                      pca=True)
  model.summary()
  ctx = ("p(a).p(aa).p(a,b).p(b).p(X).p(X,Y)."
         "q(a).q(aa).q(a,b).q(b).q(X).q(X,Y)."
         "r(a).r(aa).r(a,b).r(b).r(X).r(X,Y)."
         "s(a).s(aa).s(a,b).s(b).s(X).s(X,Y)."
         "t(a).t(aa).t(a,b).t(b).t(X).t(X,Y).")
  ctx = [r + '.' for r in ctx.split('.')[:-1]]
  embds = get_pca(ctx, model)
  plt.scatter(embds[:, 0], embds[:, 1])
  for pred, x, y in zip(ctx, embds[:, 0], embds[:, 1]):
    plt.annotate(pred, xy=(x, y), xytext=(0, 10), textcoords='offset points')
  plt.title("Predicate Embeddings")
  plt.savefig(ARGS.outf, bbox_inches='tight')

if __name__ == '__main__':
  globals()[ARGS.function]()
