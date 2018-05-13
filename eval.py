"""Evaluation module for logic-memnn"""
import argparse
from glob import glob
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
parser.add_argument("-mf", "--model_file", help="Model weights file.")
parser.add_argument("--dim", default=64, type=int, help="Latent dimension.")
parser.add_argument("-f", "--function", default="evaluate", help="Function to run.")
parser.add_argument("--outf", default="plot.png", help="Plot output file.")
parser.add_argument("-s", "--summary", action="store_true", help="Dump model summary on creation.")
parser.add_argument("-its", "--iterations", default=4, type=int, help="Number of model iterations.")
parser.add_argument("-bs", "--batch_size", default=32, type=int, help="Evaluation batch_size.")
parser.add_argument("-p", "--pad", action="store_true", help="Pad context with blank rule.")
ARGS = parser.parse_args()

MODEL_NAME = ARGS.model
MODEL_FILE = "weights/"+MODEL_NAME+str(ARGS.dim)+".h5"
if ARGS.model_file:
  MODEL_FILE = ARGS.model_file

# Stop numpy scientific printing
np.set_printoptions(suppress=True)

def create_model(**kwargs):
  """Create model from global arguments."""
  # Load in the model
  model = build_model(MODEL_NAME, MODEL_FILE,
                      char_size=len(CHAR_IDX)+1,
                      dim=ARGS.dim,
                      **kwargs)
  if ARGS.summary:
    model.summary()
  return model

def evaluate():
  """Evaluate model on each test data."""
  model = create_model(iterations=ARGS.iterations, training=True)
  for i in range(1, 13):
    dgen = LogicSeq.from_file("data/test_task{}.txt".format(i), ARGS.batch_size, pad=ARGS.pad)
    print(model.evaluate_generator(dgen))

def eval_nstep():
  """Evaluate model on nstep deduction."""
  # Load available models
  models = [(mf.split('/')[1].split('.')[0], mf)
            for mf in glob("weights/*.h5")]
  print("Found models:", models)
  # Evaluate every model on test data
  results = {m[0]:list() for m in models}
  arange = np.arange(1, 25)
  for i in arange:
    dgen = LogicSeq.from_file("data/test_nstep{}.txt".format(i), ARGS.batch_size, pad=ARGS.pad)
    for mname, mf in models:
      model = build_model(mname, mf,
                          char_size=len(CHAR_IDX)+1,
                          dim=int(mname[-2:]),
                          iterations=max(4, i+1))
      results[mname].append(model.evaluate_generator(dgen)[1])
  print("RESULTS:")
  print(results)
  # Plot the results
  for mname, rs in results.items():
    plt.plot(arange, rs, label=mname.upper())
  plt.ylim(0.4, 1.0)
  plt.ylabel("Accuracy")
  plt.xticks(arange)
  plt.xlabel("# of steps")
  plt.vlines(3, 0.4, 1.0, linestyles='dashed', label='training')
  plt.legend()
  plt.savefig(ARGS.outf, bbox_inches='tight')

def eval_len(item='pl'):
  """Evaluate model on increasing constant and predicate lengths."""
  # Load available models
  models = [(mf.split('/')[1].split('.')[0], mf)
            for mf in glob("weights/*.h5")]
  print("Found models:", models)
  results = {m[0]:list() for m in models}
  models = [(mname, build_model(mname[:-2], mf, char_size=len(CHAR_IDX)+1, dim=int(mname[-2:])))
            for mname, mf in models]
  # Evaluate every model on test data
  arange = np.arange(1, 33)
  for i in arange:
    dgen = LogicSeq.from_file("data/test_{}{}.txt".format(item, i), ARGS.batch_size, pad=ARGS.pad)
    for mname, m in models:
      results[mname].append(m.evaluate_generator(dgen)[1])
  print("RESULTS:")
  print(results)
  # Plot the results
  for mname, rs in results.items():
    plt.plot(arange, rs, label=mname.upper())
  plt.ylim(0.4, 1.0)
  plt.ylabel("Accuracy")
  plt.xticks(arange)
  if item == 'pl':
    plt.xlabel("Length of predicates (characters)")
  else:
    plt.xlabel("Length of constants (characters)")
  plt.vlines(2, 0.4, 1.0, linestyles='dashed', label='training')
  plt.legend()
  plt.savefig(ARGS.outf, bbox_inches='tight')

def eval_pred_len():
  """Evaluate model on increasing predicate lengths."""
  eval_len(item='pl')

def eval_const_len():
  """Evaluate model on increasing constant lengths."""
  eval_len(item='cl')

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

def offset(x):
  """Calculate offset for annotation."""
  r = np.random.randint(10, 30)
  return -r if x > 0 else r

def plot_single_preds():
  """Plot embeddings of single character predicates."""
  model = create_model(pca=True)
  syms = "abcdefghijklmnopqrstuvwxyz"
  ctx, splits = list(), list()
  for p in syms:
    for c in syms:
      ctx.append("{}({}).".format(p,c))
    splits.append(len(ctx))
  embds = get_pca(ctx, model)
  prev_sp = 0
  for sp in splits:
    plt.scatter(embds[prev_sp:sp, 0], embds[prev_sp:sp, 1])
    pred, x, y = ctx[prev_sp][0]+"(?)", embds[prev_sp, 0], embds[prev_sp, 1]
    prev_sp = sp
    xf, yf = offset(x), offset(y)
    plt.annotate(pred, xy=(x, y), xytext=(xf, yf), textcoords='offset points', arrowprops={'arrowstyle': '-'})
  plt.savefig(ARGS.outf, bbox_inches='tight')

def plot_pred_saturation():
  """Plot predicate embedding saturation."""
  model = create_model(pca=True)
  ctx, splits = list(), list()
  for i in range(2, 33):
    p = "".join(["p"]*i)
    ctx.append("{}(a).".format(p))
    splits.append(len(ctx))
  embds = get_pca(ctx, model)
  plt.scatter(embds[::2, 0], embds[::2, 1])
  plt.scatter(embds[1::2, 0], embds[1::2, 1])
  prev_sp = 0
  for i, sp in enumerate(splits):
    pred, x, y = ctx[prev_sp], embds[prev_sp, 0], embds[prev_sp, 1]
    count = pred.count('p')
    if count <= 6:
      xf, yf = offset(x), offset(y)
      plt.annotate(pred, xy=(x, y), xytext=(xf, yf), textcoords='offset points', arrowprops={'arrowstyle': '-'})
    elif i % 3 == 0 and count < 20 or i == len(splits)-1:
      pred = str(count)+"*p(a)"
      xf, yf = offset(x), offset(y)
      plt.annotate(pred, xy=(x, y), xytext=(xf, yf), textcoords='offset points', arrowprops={'arrowstyle': '-'})
    prev_sp = sp
  # Plot contour
  X = np.linspace(min(embds[:,0]), max(embds[:,0]), 40)
  Y = np.linspace(min(embds[:,1]), max(embds[:,1]), 40)
  X, Y = np.meshgrid(X, Y)
  Z = np.sqrt((X-embds[-1,0])**2 + (Y-embds[-1,1])**2)
  plt.contour(X, Y, Z, colors='grey')
  plt.savefig(ARGS.outf, bbox_inches='tight')

def plot_template(preds, temps):
  """Plot PCA of templates with given predicates."""
  model = create_model(pca=True)
  ctx, splits = list(), list()
  for t in temps:
    for p in preds:
      ctx.append(t.format(p))
    splits.append(len(ctx))
  embds = get_pca(ctx, model)
  prev_sp = 0
  for sp in splits:
    plt.scatter(embds[prev_sp:sp, 0], embds[prev_sp:sp, 1])
    prev_sp = sp
    pred, x, y = ctx[sp-1], embds[sp-1, 0], embds[sp-1, 1]
    xf, yf = offset(x), offset(y)
    plt.annotate(pred, xy=(x, y), xytext=(xf, yf), textcoords='offset points', arrowprops={'arrowstyle': '-'})
  plt.savefig(ARGS.outf, bbox_inches='tight')

def plot_struct_preds():
  """Plot embeddings of different structural predicates."""
  ps = ['w', 'q', 'r', 's', 't', 'v', 'u', 'p']
  temps = ["{}(X,Y).", "{}(A,A).", "{}(X).", "{}(Z).",
           "{}(a,b).", "{}(x,y).", "{}(a).", "{}(xy).",
           "-{}(a,b).", "-{}(x,y).", "-{}(a).", "-{}(xy).",
           "-{}(X,Y).", "-{}(A,A).", "-{}(X).", "-{}(Z)."]
  plot_template(ps, temps)

def plot_rules():
  """Plot embeddings of rules."""
  ps = ['w', 'a', 'b', 'c', 'd', 'e', 's', 't', 'v', 'u', 'p']
  temps = ["{}(X):-q(X).", "{}(X):--q(X).", "{}(X):-q(X);r(X).", "{}(X).",
           "{}(X,Y).", "{}(X,Y):--q(Y,X).", "{}(X,Y):--q(X,Y).",
           "{}(X,Y):-q(X,Y).", "{}(X,Y):-q(Y,X).", "{}(X,Y):-q(X);r(Y).",
           "{}(a,b).", "{}(x,y).", "{}(a).", "{}(xy).",
           "{}(X):--q(X);r(X).", "{}(X):-q(X);-r(X)."]
  plot_template(ps, temps)

def plot_attention():
  """Plot attention vector over given context."""
  model = create_model(iterations=ARGS.iterations, training=False)
  ctxs = ["p(X):-q(X).q(X):-r(X).r(X):-s(X).s(a).t(a).",
          "p(X):-q(X);r(X).r(a).q(a).r(b).t(a).",
          "p(X):-q(X).p(X):-r(X).p(b).r(a).t(a)."]
  plt.set_cmap("Blues")
  for i, ctx in enumerate(ctxs):
    print("CTX:", ctx)
    rs = ctx.split('.')[:-1]
    ctx = [r + '.' for r in rs]
    dgen = LogicSeq([(ctx, "p(a).", 0)], 1, False, False, pad=ARGS.pad)
    out = model.predict_generator(dgen)
    sims = out[:-1]
    out = np.round(np.asscalar(out[-1]), 2)
    sims = np.stack(sims, axis=0).squeeze()
    print("ATTS:", sims)
    sims = sims.T
    ax = plt.subplot(1, len(ctxs), i+1)
    ax.xaxis.tick_top()
    plt.imshow(sims)
    if ARGS.pad:
      plt.yticks(range(len(rs)+1), rs + ["$\phi$"])
    else:
      plt.yticks(range(len(rs)), rs)
    plt.xlabel("p(Q|C)=" + str(out))
    plt.xticks(range(4), range(1, 5))
    # if i == len(ctxs) - 1:
      # plt.colorbar(fraction=0.05)
    print("OUT:", out)
  plt.tight_layout()
  plt.savefig(ARGS.outf, bbox_inches='tight')

if __name__ == '__main__':
  globals()[ARGS.function]()
