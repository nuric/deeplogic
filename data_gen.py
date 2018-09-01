"""Data generation script for logic programs."""
import argparse
import random as R

# Symbol Pool
CONST_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
VAR_SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PRED_SYMBOLS = "abcdefghijklmnopqrstuvwxyz"
EXTRA_SYMBOLS = "-,()"

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

def choices(symbols, k):
  """Return k many symbols with replacement. Added in v3.6."""
  return [R.choice(symbols) for _ in range(k)]

def r_string(symbols, length):
  """Return random sequence from given symbols."""
  return ''.join(R.choice(symbols)
                 for _ in range(length))

def r_symbols(size, symbols, length, used=None):
  """Return unique random from given symbols."""
  if length == 1 and not used:
    return R.sample(symbols, size)
  rset, used = set(), set(used or [])
  while len(rset) < size:
    s = r_string(symbols, R.randint(1, length))
    if s not in used:
      rset.add(s)
  return list(rset)

def r_consts(size, used=None):
  """Return size many unique constants."""
  return r_symbols(size, CONST_SYMBOLS, ARGS.constant_length, used)

def r_vars(size, used=None):
  """Return size many unique variables."""
  return r_symbols(size, VAR_SYMBOLS, ARGS.variable_length, used)

def r_preds(size, used=None):
  """Return size many unique predicates."""
  return r_symbols(size, PRED_SYMBOLS, ARGS.predicate_length, used)

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

def gen_task(context, targets, upreds):
  """Fill context with random preds and output program."""
  # Fill with random rules up to certain task
  ctx = context.copy() # Don't modify the original context
  for _ in range(ARGS.noise_size):
    task = "gen_task" + str(R.randint(1, max(1, ARGS.task)))
    ctx.append(globals()[task](upreds))
  output(ctx, targets)

def add_pred(context, pred, upreds, uconsts, psuccess=0.0):
  """Fail a ground case predicate given context."""
  # Maybe succeed by adding to context
  if R.random() < psuccess:
    context.append([pred])
  if R.random() < 0.5:
    # The constant doesn't match
    args = pred[1].copy()
    args[R.randrange(len(args))] = r_consts(1, uconsts)[0]
    context.append([(pred[0], args)])
  if R.random() < 0.5:
    # The predicate doesn't match
    p = r_preds(1, upreds)[0]
    upreds.append(p)
    context.append([(p, pred[1])])
  # The predicate doesn't appear at all

def gen_task1(upreds=None):
  """Ground instances only: p(a).q(c,b)."""
  # One or two argument predicate
  preds = r_preds(2, upreds)
  args = r_consts(R.randint(1, 2))
  rule = [(preds[0], args)]
  if upreds:
    return rule
  ctx = list()
  add_pred(ctx, rule[0], preds, args, 1.0)
  # Successful case when query appears in context
  targets = [(rule[0], 1)]
  # Fail case
  args = r_consts(R.randint(1, 2))
  fpred = (preds[1], args)
  add_pred(ctx, fpred, preds, args)
  targets.append((fpred, 0))
  gen_task(ctx, targets, preds)

def gen_task2(upreds=None):
  """Variablised facts only: p(X).q(X,Y)."""
  preds = r_preds(2, upreds)
  ctx, targets = list(), list()
  if R.random() < 0.5:
    # Double variable same argument
    v = r_vars(1)[0]
    rule = [(preds[0], [v, v])]
    if upreds:
      return rule
    ctx.append(rule)
    # Successful double variable grounding
    cs = r_consts(2)
    c = R.choice(cs)
    targets.append(((preds[0], [c, c]), 1))
    # Fail on non-unique variable grounding
    targets.append(((preds[0], cs), 0))
  else:
    # Double variable different argument
    # Single variable argument
    argc = R.randint(1, 2)
    args = r_vars(argc)
    rule = [(preds[0], args)]
    if upreds:
      return rule
    ctx.append(rule)
    # Successful unique argument grounding
    args = choices(r_consts(2), argc)
    targets.append(((preds[0], args), 1))
    # Fail on out of context predicate with same arguments
    targets.append(((preds[1], args), 0))
  gen_task(ctx, targets, preds)

def nstep_deduction(steps, negation=False, upreds=None):
  assert steps >= 1, "Need at least 1 step deduction."
  preds = r_preds(2 if upreds else 3+steps, upreds)
  consts = r_consts(2)
  ctx, targets = list(), list()
  prefix = '-' if negation else ''
  if R.random() < 0.5:
    # Double variable swap deduction rules
    vs = r_vars(2)
    rule = [(preds[0], vs), (prefix+preds[1], vs[::-1])]
    if upreds:
      return rule
    ctx.append(rule)
    # Add the n steps
    swapc = 1
    for j in range(steps-1):
      vs = r_vars(2)
      toswap = R.random() < 0.5 # Do we swap again?
      args = vs[::-1] if toswap else vs
      ctx.append([(preds[j+1], vs), (preds[j+2], args)])
      swapc += int(toswap)
    # Add the ground case
    args = r_consts(2)
    add_pred(ctx, (preds[steps], args), preds, consts, 1.0)
    args = args if swapc % 2 == 0 else args[::-1]
    targets.append(((preds[0], args), 1-int(negation)))
    targets.append(((preds[0], args[::-1]), int(negation)))
    gen_task(ctx, targets, preds)
  else:
    # Double variable non-swap deduction rules
    # Single variable deduction rules
    argc = R.randint(1, 2)
    vs = r_vars(argc)
    rule = [(preds[0], vs), (prefix+preds[1], vs)]
    if upreds:
      return rule
    ctx.append(rule)
    # Add the n steps
    for j in range(steps-1):
      vs = r_vars(argc)
      ctx.append([(preds[j+1], vs), (preds[j+2], vs)])
    args = choices(r_consts(2), argc)
    # Add the ground case
    cctx = ctx.copy()
    spred = (preds[steps], args)
    add_pred(cctx, spred, preds, args, 1.0)
    targets = [((preds[0], args), 1-int(negation))]
    gen_task(cctx, targets, preds)
    # Add failure case
    if R.random() < 0.5:
      # Fail on broken chain
      p = r_preds(1, preds)[0]
      preds.append(p)
      add_pred(ctx, spred, preds, args, 1.0)
      ctx[0] = [(preds[0], vs), (prefix+p, vs)]
    else:
      # Fail on last ground case
      add_pred(ctx, spred, preds, args)
    targets = [((preds[0], args), int(negation))]
    gen_task(ctx, targets, preds)

def gen_task3(upreds=None):
  """Single step deduction: p(X):-q(X)."""
  return nstep_deduction(1, upreds=upreds)

def gen_task4(upreds=None):
  """Double step deduction: p(X):-q(X).q(X):-r(X)."""
  return nstep_deduction(2, upreds=upreds)

def gen_task5(upreds=None):
  """Triple step deduction."""
  return nstep_deduction(3, upreds=upreds)

def logical_and(negation=False, upreds=None):
  """Logical AND with optional negation: p(X):-q(X);r(X)."""
  preds = r_preds(3, upreds)
  argc = R.randint(1, 2)
  # Double variable AND with different vars
  # Single variable AND
  vs = r_vars(argc)
  rule = [(preds[0], vs),
          (preds[1], vs[:1]),
          (preds[2], vs[1:] or vs)]
  if upreds:
    return rule
  ctx = [rule]
  # Create the ground arguments
  args = choices(r_consts(2), argc)
  prem1 = (preds[1], args[:1])
  prem2 = (preds[2], args[1:] or args)
  prems = [prem1, prem2]
  if negation:
    # Add negation to random predicate in body
    ridx = R.randrange(2)
    p, pargs = ctx[-1][ridx+1]
    ctx[-1][ridx+1] = ('-' + p, pargs)
    # Successful case when negation fails
    cctx = ctx.copy()
    add_pred(cctx, prems[ridx], preds, args)
    cctx.append([prems[1-ridx]])
    targets = [((preds[0], args), 1)]
    gen_task(cctx, targets, preds)
    # Fail one premise randomly
    fidx = R.randrange(2)
    if ridx == fidx:
      # To fail negation add ground instance
      ctx.append([prems[ridx]])
      # Succeed other with some probability
      add_pred(ctx, prems[1-ridx], preds, args, 0.8)
    else:
      # Fail non-negated premise
      add_pred(ctx, prems[1-ridx], preds, args)
      # Still succeed negation
      add_pred(ctx, prems[ridx], preds, args)
    targets = [((preds[0], args), 0)]
    gen_task(ctx, targets, preds)
  else:
    # Create successful context
    cctx = ctx.copy()
    add_pred(cctx, prems[0], preds, args, 1.0)
    add_pred(cctx, prems[1], preds, args, 1.0)
    targets = [((preds[0], args), 1)]
    gen_task(cctx, targets, preds)
    # Fail one premise randomly
    fidx = R.randrange(2)
    add_pred(ctx, prems[fidx], preds, args)
    # Succeed the other with some probability
    add_pred(ctx, prems[1-fidx], preds, args, 0.8)
    targets = [((preds[0], args), 0)]
    gen_task(ctx, targets, preds)

def gen_task6(upreds=None):
  """Logical AND: p(X):-q(X);r(X)."""
  return logical_and(upreds=upreds)

def logical_or(negation=False, upreds=None):
  """Logical OR with optional negation: p(X):-q(X).p(X):-r(X)."""
  preds = r_preds(3, upreds)
  # Double or single variable OR
  argc = R.randint(1, 2)
  vs = r_vars(argc)
  swap = R.random() < 0.5
  prefix = '-' if negation else ''
  rule = [(preds[0], vs), (prefix + preds[1], vs[::-1] if swap else vs)]
  if upreds:
    return rule
  ctx = list()
  ctx.append(rule)
  # Add the extra branching rules
  ctx.append([(preds[0], vs), (preds[2], vs)])
  args = r_consts(argc)
  ctx.append([(preds[0], args)])
  if swap and argc == 2:
    args = r_consts(argc, args)
    add_pred(ctx, (preds[1], args), preds, args, 1.0)
    args = args[::-1] if swap else args
    targets = [((preds[0], args), 1-int(negation)),
               ((preds[0], args[::-1]), int(negation))]
    gen_task(ctx, targets, preds)
  elif not negation and R.random() < 0.2:
    # Sneaky shorcut case
    targets = [((preds[0], args), 1)]
    gen_task(ctx, targets, preds)
    del ctx[-1]
    targets = [((preds[0], args), 0)]
    gen_task(ctx, targets, preds)
  else:
    # Succeed either from them
    prems = [(preds[i], r_consts(argc, args)) for i in range(1, 3)]
    sidx = R.randrange(2)
    cctx = ctx.copy()
    if negation and sidx == 0:
      # Succeed by failing negation
      add_pred(cctx, prems[0], preds, prems[0][1])
      # Possibly succeed other prem
      add_pred(cctx, prems[1], preds, prems[1][1], 0.2)
    else:
      # Succeed by adding ground case
      add_pred(cctx, prems[sidx], preds, prems[sidx][1], 1.0)
      # Possibly succeed other prem
      add_pred(cctx, prems[1-sidx], preds, prems[1-sidx][1], 0.2)
    targets = [((preds[0], prems[sidx][1]), 1)]
    gen_task(cctx, targets, preds)
    # Fail both
    add_pred(ctx, prems[0], preds, prems[0][1], int(negation))
    add_pred(ctx, prems[1], preds, prems[1][1])
    targets = [((preds[0], prems[sidx][1]), 0)]
    gen_task(ctx, targets, preds)


def gen_task7(upreds=None):
  """Logical OR: p(X):-q(X).p(X):-r(X)."""
  return logical_or(upreds=upreds)

def gen_task8(upreds=None):
  """Transitive case: p(X,Y):-q(X,Z);r(Z,Y)."""
  preds = r_preds(3, upreds)
  # Existential variable with single choice
  vs = r_vars(3)
  rule = [(preds[0], [vs[0], vs[2]]),
          (preds[1], vs[:2]),
          (preds[2], vs[1:])]
  if upreds:
    return rule
  ctx = [rule]
  # Add matching ground cases
  args = r_consts(3)
  add_pred(ctx, (preds[1], args[:2]), preds, args, 1.0)
  add_pred(ctx, (preds[2], args[1:]), preds, args, 1.0)
  # Add non-matching ground cases
  argso = r_consts(3)
  argso.insert(R.randint(1, 2), r_consts(1, argso)[0])
  add_pred(ctx, (preds[1], argso[:2]), preds, argso, 1.0)
  add_pred(ctx, (preds[2], argso[2:]), preds, argso, 1.0)
  # Successful case
  # Fail on half-matching existential
  targets = [((preds[0], [args[0], args[2]]), 1),
             ((preds[0], [argso[0], argso[3]]), 0)]
  gen_task(ctx, targets, preds)

def gen_task9(upreds=None):
  """Single step deduction with NBF: p(X):--q(X)."""
  return nstep_deduction(1, True, upreds)

def gen_task10(upreds=None):
  """Double step deduction with NBF: p(X):--q(X).q(X):-r(X)."""
  return nstep_deduction(2, True, upreds)

def gen_task11(upreds=None):
  """Logical AND with NBF: p(X):-q(X);-r(X)."""
  return logical_and(True, upreds)

def gen_task12(upreds=None):
  """Logical OR with NBF: p(X):--q(X).p(X):-r(X)."""
  return logical_or(True, upreds)

def gen_task0():
  """Generate an ILP task example."""
  argc = 1
  goal= 'f'
  premise = 'b'
  ctx, targets = list(), list()
  # Generate according to goal <- premise
  args = r_consts(argc)
  # Add the successful ground case
  ctx.append([(premise, args)])
  targets.append(((goal, args), 1))
  # Fail on non-matching constant
  args = args.copy()
  args[R.randrange(len(args))] = r_consts(1, args)[0]
  preds = r_preds(3)
  ctx.append([(preds[0], args)])
  targets.append(((goal, args), 0))
  # Add padding length dummy rule
  vs = r_vars(argc)
  ctx.append([(preds[1], vs), (preds[2], vs)])
  preds.extend([goal, premise])
  gen_task(ctx, targets, preds)

if __name__ == '__main__':
  # pylint: disable=line-too-long
  # Arguments
  parser = argparse.ArgumentParser(description="Generate logic program data.")
  parser.add_argument("-t", "--task", default=1, type=int, help="The task to generate.")
  parser.add_argument("-s", "--size", default=1, type=int, help="Number of programs to generate.")
  # Configuration parameters
  parser.add_argument("-ns", "--noise_size", default=2, type=int, help="Size of added noise rules.")
  parser.add_argument("-cl", "--constant_length", default=1, type=int, help="Length of constants.")
  parser.add_argument("-vl", "--variable_length", default=1, type=int, help="Length of variables.")
  parser.add_argument("-pl", "--predicate_length", default=1, type=int, help="Length of predicates.")
  parser.add_argument("-sf", "--shuffle_context", action="store_true", help="Shuffle context before output.")
  # Task specific options
  parser.add_argument("--nstep", type=int, help="Generate nstep deduction programs.")
  ARGS = parser.parse_args()

  # Generate given task
  task = "gen_task" + str(ARGS.task)
  for _ in range(ARGS.size):
    if ARGS.nstep:
      nstep_deduction(ARGS.nstep)
    else:
      globals()[task]()
