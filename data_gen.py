"""Data generation script for logic programs."""
import argparse
import random as R

# Symbol Pool
CONST_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
VAR_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRED_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
EXTRA_SYMBOLS = ".:-,;()"

CHARS = sorted(list(set(CONST_SYMBOLS+VAR_SYMBOLS+PRED_SYMBOLS+EXTRA_SYMBOLS)))
# Reserve 0 for padding
CHAR_IDX = dict((c, i+1) for i, c in enumerate(CHARS))
IDX_CHAR = [0]
IDX_CHAR.extend(CHARS)

# Predicate Templates
FACT_T = "{}."
RULE_T = "{}:-{}."
PRED_T = "{}({})"
ARG_SEP = ','
PRED_SEP = ';'
TARGET_T = "? {} {}"

def r_string(symbols, length):
  """Return random sequence from given symbols."""
  return ''.join(R.choice(symbols)
                 for _ in range(length))

def r_symbols(size, symbols, length):
  """Return unique random from given symbols."""
  if length == 1:
    return R.sample(symbols, size)
  rset = set()
  while len(rset) < size:
    rset.add(r_string(symbols, length))
  return list(rset)

def r_consts(size):
  """Return size many unique constants."""
  return r_symbols(size, CONST_SYMBOLS, ARGS.constant_length)

def r_vars(size):
  """Return size many unique variables."""
  return r_symbols(size, VAR_SYMBOLS, ARGS.variable_length)

def r_preds(size):
  """Return size many unique predicates."""
  return r_symbols(size, PRED_SYMBOLS, ARGS.predicate_length)

def write_p(pred):
  """Format single predicate tuple into string."""
  return PRED_T.format(pred[0], ARG_SEP.join(pred[1]))

def write_r(preds):
  """Convert rule predicate tuple into string."""
  head = write_p(preds[0])
  # Is it just a fact
  if len(preds) == 1:
    return FACT_T.format(head)
  # We have a rule
  return RULE_T.format(head, PRED_SEP.join([write_p(p) for p in preds[1:]]))

def output(context, targets):
  """Print the context and given targets."""
  # context: [[('p', ['a', 'b'])], ...]
  # targets: [(('p', ['a', 'b']), 1), ...]
  if ARGS.shuffle_context:
    R.shuffle(context)
  print('\n'.join([write_r(c) for c in context]))
  for t, v in targets:
    print(TARGET_T.format(write_r([t]), v))

def gen_task1(ctx_size):
  """Ground instances only: p(a).q(c,b)."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  # Create context with both single and double arguments
  ctx = list()
  for i in range(ctx_size):
    if R.random() < 0.5:
      args = [R.choice(consts[:-1]), R.choice(consts[:-1])]
    else:
      args = [R.choice(consts[:-1])]
    ctx.append([(preds[i], args)])
  # Successful case when query appears in context
  targets = [(ctx[0][0], 1)]
  # What are possible failures
  fails = list()
  # Out of context constant fails
  pred = R.choice(ctx)[0]
  args = pred[1].copy()
  args[R.randrange(len(args))] = consts[-1]
  fails.append(((pred[0], args), 0))
  # Out of context predicate fails
  fails.append(((preds[-1], [R.choice(consts[:-1])]), 0))
  targets.append(R.choice(fails))
  output(ctx, targets)

def gen_task2(ctx_size):
  """Variablised facts only: p(X).q(X,Y)."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, div = list(), ctx_size//4
  # Double variable same argument
  for i in range(div):
    v = R.choice(var)
    ctx.append([(preds[i], [v, v])])
  # Double variable unique argument
  for i in range(div, div*2):
    args = R.sample(var, 2)
    ctx.append([(preds[i], args)])
  # Single variable argument
  for i in range(div*2, div*3):
    ctx.append([(preds[i], [R.choice(var)])])
  # Some ground instances
  for i in range(div*3, ctx_size):
    ctx.append([(preds[i], [R.choice(consts)])])
  targets = list()
  # Successful double variable grounding
  p = ctx[div][0][0]
  targets.append(((p, [R.choice(consts), R.choice(consts)]), 1))
  # Successful single variable grounding
  p = ctx[div*2][0][0]
  targets.append(((p, [R.choice(consts)]), 1))
  # Fail on non-unique variable grounding
  p = ctx[0][0][0]
  targets.append(((p, R.sample(consts, 2)), 0))
  # Out of context predicate fails
  targets.append(((preds[-1], [R.choice(consts[:-1])]), 0))
  output(ctx, targets)

def gen_task3(ctx_size):
  """Single step deduction: p(X):-q(X)."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, div = list(), ctx_size//2
  # Variable deduction rules
  for i in range(div):
    v = R.choice(var)
    ctx.append([(preds[i*2], [v]), (preds[i*2+1], [v])])
  # Ground instances
  for i in range(div):
    ctx.append([(preds[i*2+1], [consts[i]])])
  targets = list()
  # Successful deduction
  p = ctx[0][0][0]
  targets.append(((p, [consts[0]]), 1))
  p = ctx[1][0][0]
  targets.append(((p, [consts[1]]), 1))
  # Fail on unknown const deduction
  p = ctx[div-1][0][0]
  targets.append(((p, [R.choice(consts[div:])]), 0))
  # Fail on unsatisfied premise
  p = ctx[1][0][0]
  targets.append(((p, [R.choice(consts[div:])]), 0))
  output(ctx, targets)

def gen_task4(ctx_size):
  """Transitive case: p(X,Y):-q(X,Z);R(Z,Y)."""
  preds = r_preds(ctx_size+1)
  consts = r_consts(ctx_size+1)
  var = r_vars(ctx_size)
  ctx, div = list(), ctx_size//3
  # Transitive rules
  for i in range(div):
    vs = R.sample(var, 3)
    r = [(preds[i*3], [vs[0], vs[2]]),
         (preds[i*3+1], [vs[0], vs[1]]),
         (preds[i*3+2], [vs[1], vs[2]])]
    ctx.append(r)
  # Ground instances to satisfy deduction
  for i in range(div//2):
    ctx.append([(preds[i*3+1], [consts[i*2], consts[i*2+1]])])
    ctx.append([(preds[i*3+2], [consts[i*2+1], consts[i*2+2]])])
  for i in range(div//2, div):
    ctx.append([(preds[i*3+1], [consts[i*2], consts[i*2+1]])])
    ctx.append([(preds[i*3+2], [consts[-1], consts[i*2+2]])])
  targets = list()
  # Successful deduction
  p = ctx[0][0][0]
  targets.append(((p, [consts[0], consts[2]]), 1))
  # Fail on non-matching premise
  p = ctx[div//2][0][0]
  targets.append(((p, [consts[div], consts[div+2]]), 0))
  output(ctx, targets)

if __name__ == '__main__':
  # Arguments
  parser = argparse.ArgumentParser(description="Generate logic program data.")
  parser.add_argument("task", help="The task to generate.")
  parser.add_argument("size", type=int, help="Number of programs to generate.")
  parser.add_argument("-cs", "--context_size", default=6, type=int, help="Size of program context.")
  parser.add_argument("-cl", "--constant_length", default=1, type=int, help="Length of constants.")
  parser.add_argument("-vl", "--variable_length", default=1, type=int, help="Length of variables.")
  # pylint: disable=line-too-long
  parser.add_argument("-pl", "--predicate_length", default=1, type=int, help="Length of predicates.")
  parser.add_argument("-s", "--shuffle_context", action="store_true", help="Shuffle context before output.")
  ARGS = parser.parse_args()

  # Generate given task
  task = "gen_task" + ARGS.task
  for _ in range(ARGS.size):
    globals()[task](ARGS.context_size)
