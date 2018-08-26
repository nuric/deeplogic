"""Evaluation module for logic-memnn"""
import argparse
from glob import glob
import numpy as np
from sklearn.decomposition import PCA

from data_gen import CHAR_IDX
from utils import LogicSeq
from models import build_model

# Arguments
parser = argparse.ArgumentParser(description="Evaluate logic-memnn models.")
parser.add_argument("model", help="The name of the module to train.")
parser.add_argument("model_file", help="Model filename.")
parser.add_argument("-md", "--model_dir", help="Model weights directory ending with /.")
parser.add_argument("--dim", default=64, type=int, help="Latent dimension.")
parser.add_argument("-f", "--function", default="evaluate", help="Function to run.")
parser.add_argument("--outf", help="Plot to output file instead of rendering.")
parser.add_argument("-s", "--summary", action="store_true", help="Dump model summary on creation.")
parser.add_argument("-its", "--iterations", default=4, type=int, help="Number of model iterations.")
parser.add_argument("-bs", "--batch_size", default=32, type=int, help="Evaluation batch_size.")
parser.add_argument("-p", "--pad", action="store_true", help="Pad context with blank rule.")
ARGS = parser.parse_args()

if ARGS.outf:
  import matplotlib
  matplotlib.use("Agg") # Bypass X server
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_NAME = ARGS.model
MODEL_FNAME = ARGS.model_file
MODEL_WF = (ARGS.model_dir or "weights/") + MODEL_FNAME + '.h5'

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

def evaluate():
  """Evaluate model on each test data."""
  model = create_model(iterations=ARGS.iterations, training=True)
  for s in ['val', 'easy', 'med', 'hard']:
    results = list()
    for i in range(1, 13):
      dgen = LogicSeq.from_file("data/test_{}_task{}.txt".format(s,i), ARGS.batch_size, pad=ARGS.pad, verbose=False)
      _, acc = model.evaluate_generator(dgen) # [loss, acc]
      results.append(acc)
    print(s.upper(), ','.join(map(str, results)), "MEAN:", np.mean(results, axis=0))

def showsave_plot():
  """Show or save plot."""
  if ARGS.outf:
    plt.savefig(ARGS.outf, bbox_inches='tight')
  else:
    plt.show()

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
      model = build_model(mname[:-2], mf,
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
  plt.xlim(1, arange[-1])
  plt.xticks(arange[::2])
  plt.xlabel("# of steps")
  plt.vlines(3, 0.4, 1.0, colors='grey', linestyles='dashed', label='training')
  plt.legend()
  showsave_plot()

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
  arange = np.arange(2, 65)
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
  plt.xlim(2, arange[-1])
  plt.xticks(arange[::8])
  if item == 'pl':
    plt.xlabel("Length of predicates (characters)")
  else:
    plt.xlabel("Length of constants (characters)")
  plt.legend()
  showsave_plot()

def eval_pred_len():
  """Evaluate model on increasing predicate lengths."""
  eval_len(item='pl')

def eval_const_len():
  """Evaluate model on increasing constant lengths."""
  eval_len(item='cl')

def get_pca(context, model, dims=2):
  """Plot the PCA of predicate embeddings."""
  dgen = LogicSeq([[(context, "z(z).", 0)]], 1,
                  train=False, shuffle=False, zeropad=False)
  embds = model.predict_generator(dgen)
  embds = embds.squeeze()
  pca = PCA(dims)
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
  preds = list("pqrv")
  preds.extend([''.join([e]*2) for e in preds])
  for p in preds:
    for c in syms:
      ctx.append("{}({}).".format(p,c))
    splits.append(len(ctx))
  embds = get_pca(ctx, model, dims=len(preds))
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  prev_sp = 0
  for sp in splits:
    x, y, z = embds[prev_sp:sp, 0], embds[prev_sp:sp, 1], embds[prev_sp:sp, -1]
    ax.scatter(x, y, z, depthshade=False)
    for i in map(syms.index, "fdgm"):
      ax.text(x[i], y[i], z[i], ctx[prev_sp+i])
    prev_sp = sp
  showsave_plot()

def plot_pred_saturation():
  """Plot predicate embedding saturation."""
  model = create_model(pca=True)
  ctx, splits = list(), list()
  for i in range(2, 65):
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
    elif i % 3 == 0 and i < 50 or i == len(splits)-1 or i == len(splits)-2:
      pred = str(count)+"*p(a)"
      xf, yf = offset(x), offset(y)
      plt.annotate(pred, xy=(x, y), xytext=(xf, yf), textcoords='offset points', arrowprops={'arrowstyle': '-'})
    prev_sp = sp
  # Plot contour
  plt.xlim(-2, 2)
  xmin, xmax = plt.xlim()
  X = np.linspace(xmin, xmax, 40)
  ymin, ymax = plt.ylim()
  Y = np.linspace(ymin, ymax, 40)
  X, Y = np.meshgrid(X, Y)
  Z = np.sqrt((X-embds[-1,0])**2 + (Y-embds[-1,1])**2)
  plt.contour(X, Y, Z, colors='grey', alpha=0.2, linestyles='dashed')
  Z = np.sqrt((X-embds[-2,0])**2 + (Y-embds[-2,1])**2)
  plt.contour(X, Y, Z, colors='grey', alpha=0.2, linestyles='dashed')
  showsave_plot()

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
  showsave_plot()

def plot_struct_preds():
  """Plot embeddings of different structural predicates."""
  ps = ['w', 'q', 'r', 's', 't', 'v', 'u', 'p']
  temps = ["{}(X,Y).", "{}(X,X).", "{}(X).", "{}(Z).",
           "{}(a,b).", "{}(x,y).", "{}(a).", "{}(xy).",
           "-{}(a,b).", "-{}(x,y).", "-{}(a).", "-{}(xy).",
           "-{}(X,Y).", "-{}(X,X).", "-{}(X).", "-{}(Z)."]
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
  ctxs = ["p(X):-q(X).q(X):-r(X).r(X):-s(X).s(a).s(b).",
          "p(X):-q(X);r(X).r(a).q(a).r(b).q(b).",
          "p(X):-q(X).p(X):-r(X).p(b).r(a).q(b)."]
  fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4))
  for i, ctx in enumerate(ctxs):
    print("CTX:", ctx)
    rs = ctx.split('.')[:-1]
    ctx = [r + '.' for r in rs]
    dgen = LogicSeq([[(ctx, "p(a).", 0)]], 1, False, False, pad=ARGS.pad)
    out = model.predict_generator(dgen)
    sims = out[:-1]
    out = np.round(np.asscalar(out[-1]), 2)
    sims = np.stack(sims, axis=0).squeeze()
    print("ATTS:", sims)
    sims = sims.T
    ticks = (["()"] if ARGS.pad else []) + ["$\phi$"]
    axes[i].get_xaxis().set_ticks_position('top')
    sns.heatmap(sims, vmin=0, vmax=1, cmap="Blues", yticklabels=rs + ticks,
                linewidths=0.5, square=True, cbar=False, ax=axes[i])
    axes[i].set_xlabel("p(Q|C)=" + str(out))
    print("OUT:", out)
  plt.tight_layout()
  showsave_plot()

if __name__ == '__main__':
  globals()[ARGS.function]()
